from imports import *
from description import *
from helper_data_structures import *


device = torch.device("cuda" if torch.cuda.is_available() else "cuda")



class task2_pipeline:
    def __init__(self, device, cifar_classifier, clip_model, vlm_path, artifact_classification, class_desc_sub, sub_class_features_dict, feature_to_class, p_threshold = 0.2):
        self.device = device
        self.cifar_classifier = pipeline("image-classification", model = cifar_classifier, device = device)
        self.clip_model, self.clip_preprocess = create_model_from_pretrained(model_name = clip_model, pretrained = "webli")
        self.clip_tokenizer = get_tokenizer(clip_model)
        self.clip_model.to(device)
        self.qwen_processor = AutoProcessor.from_pretrained(vlm_path)
        self.qwen2VL = Qwen2VLForConditionalGeneration.from_pretrained(vlm_path).to(device)
        self.artifact_classification = artifact_classification
        self.class_desc_sub = class_desc_sub
        self.sub_class_features_dict = sub_class_features_dict
        self.feature_to_class = feature_to_class
        self.p_threshold = p_threshold
        self.class_names_broad = list(artifact_classification.keys())
        self.class_names_sub = []
        for k in artifact_classification:
          self.class_names_sub.extend(artifact_classification[k])
        self.max_length = 200
        sample_encoding = self.encode_text("sample text")
        self.embedding_dim = sample_encoding.shape[1]
        self.sub_class_classifier_initial = self.sub_class_classifier_initial()
        self.results = pd.DataFrame(columns = ['image_path', 'preds'])

    def sub_class_classifier_initial(self):
        weights_sub_classes = {}
        for broad_class in self.class_names_broad:
            weights_sub_class = []
            for sub_class in self.artifact_classification[broad_class]:
                sub_class_name_embedding = self.encode_text(sub_class).to(self.device)
                template_embedding = self.encode_text(f"AI generated image containing {sub_class} as artifact")
                sub_class_desc_embedding = torch.zeros((1, self.embedding_dim))
                sub_class_desc_embedding = sub_class_desc_embedding.to(self.device)
                for desc in self.class_desc_sub[sub_class]:
                    sub_class_desc_embedding += self.encode_text(desc)
                sub_class_desc_embedding /= len(self.class_desc_sub[sub_class])
                sub_class_embedding = sub_class_name_embedding + template_embedding + sub_class_desc_embedding
                normalized_sub_class_embedding = sub_class_embedding / torch.norm(sub_class_embedding)
                weights_sub_class.append(torch.squeeze(normalized_sub_class_embedding))
            weights_sub_classes[broad_class] = weights_sub_class
        sub_class_classifier_initial = {
            "weights": weights_sub_classes,
            "class_names_sub": self.class_names_sub
        }
        return sub_class_classifier_initial

    def cifar_class(self, image_path):
        img = Image.open(image_path)
        return self.cifar_classifier(img)[0]["label"]

    def vlm_prediction(self, prompt, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path}"
                        \
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ]

        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        generated_ids = self.qwen2VL.generate(**inputs, max_new_tokens=100, temperature = 0.99)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = " ".join(output_text)
        result = result.strip()
        return result

    def encode_text(self, text):
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            tokens = self.clip_tokenizer(text).to(self.device)
            embd = self.clip_model.encode_text(tokens)
            embd/=torch.norm(embd)
            if len(embd.shape) == 1:
                embd = embd.unsqueeze(0)
        # print(embd)
        # print(type(embd))
        return embd

    def encode_image(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            embd = self.clip_model.encode_image(image_input)
            if len(embd.shape) == 1:
                embd = embd.unsqueeze(0)
            embd = embd / embd.norm(dim=-1, keepdim=True)
            return embd

    def classify_broad(self, image_path, cifar_class):
        broad_classification_p = f"This image of a/an {cifar_class} is AI generated. An AI generated image can contain artifacts or defects from the classes: {[[i] for i in self.class_names_broad]}. Output only the class names that are present in this image. DO NOT OUTPUT explainations or artifacts that are not present."
        broad_prediction = self.vlm_prediction(broad_classification_p, image_path)
        preds = broad_prediction.split(', ')
        preds = [i.lower() for i in list(set(preds))]
        if cifar_class in living:
            preds.append("biological and anatomical issues")
        return preds

    def sub_class_probs_initial(self, image_path, broad_class_preds):
        possible_sub_classes = []
        for broad_class in broad_class_preds:
            if broad_class not in artifact_classification:
              continue
            possible_sub_classes.extend(self.artifact_classification[broad_class])
        image_feature = self.encode_image(image_path)
        image_feature /= torch.norm(image_feature)
        prediction_feature = self.encode_text(possible_sub_classes)
        prediction_feature /= torch.norm(prediction_feature)
        query_feature = image_feature + prediction_feature
        sub_class_classifier_initial_weights = []
        for broad_class in self.class_names_broad:
            if broad_class in broad_class_preds:
                if broad_class not in artifact_classification:
                  continue
                sub_class_classifier_initial_weights.extend(self.sub_class_classifier_initial["weights"][broad_class])
        sub_class_classifier_initial_weights = torch.stack(sub_class_classifier_initial_weights).t()
        confidence_scores = torch.matmul(query_feature, sub_class_classifier_initial_weights)[0]
        temp_probs = torch.nn.functional.softmax(confidence_scores)
        sub_class_probs_initial = {}
        for i in range(len(possible_sub_classes)):
            sub_class_probs_initial[possible_sub_classes[i]] = temp_probs[i].item()
        for i in sub_class_probs_initial:
            if i in artifact_classification['lighting and reflection issues']:
                sub_class_probs_initial[i]/=1.5
        for i in sub_class_probs_initial:
            if i in artifact_classification['biological and anatomical issues']:
                sub_class_probs_initial[i]/=1.4
        return sub_class_probs_initial

    def sub_class_feature_probs(self, image_path, broad_class_preds):
        image_embedding = self.encode_image(image_path)
        image_embedding /= torch.norm(image_embedding)
        caption_embeddings = []
        possible_sub_classes = []
        possible_sub_class_features = []
        for broad_class in self.class_names_broad:
            if broad_class in broad_class_preds:
                if broad_class not in artifact_classification:
                  continue
                possible_sub_classes.extend(self.artifact_classification[broad_class])
        for sub_class in possible_sub_classes:
            possible_sub_class_features.extend(self.sub_class_features_dict[sub_class])
        for sub_class_feature in possible_sub_class_features:
            caption_prompt = f"A {sub_class_feature} as artifact"
            caption_embeddings.extend(self.encode_text(caption_prompt))
        # print(caption_embeddings)
        caption_embeddings = torch.stack(caption_embeddings).t()
        temp_probs = torch.matmul(image_embedding[0], caption_embeddings)
        sub_class_feature_probs = {}
        for i in range(len(possible_sub_class_features)):
            sub_class_feature_probs[possible_sub_class_features[i]] = temp_probs[i].item()
        return sub_class_feature_probs

    def combine_probs(self, broad_class_preds, sub_class_probs_initial, sub_class_feature_probs):
        possible_sub_classes = []
        for broad_class in self.class_names_broad:
            if broad_class in broad_class_preds:
                if broad_class not in artifact_classification:
                  continue
                possible_sub_classes.extend(self.artifact_classification[broad_class])
        combined_sub_class_feature_confidence = {}
        for sub_class in possible_sub_classes:
            sub_class_features = sub_class_features_dict[sub_class]
            for feature in sub_class_features:
                combined_sub_class_feature_confidence[feature] = sub_class_feature_probs[feature]*sub_class_probs_initial[sub_class]
        return combined_sub_class_feature_confidence

    def final_predictions(self, cifar_class, combined_sub_class_feature_confidence):
        final_sub_class_confidence = {}
        for feature in combined_sub_class_feature_confidence:
            # print(feature)
            # print(feature_to_class[feature])
            sub_class = self.feature_to_class[feature]
            if sub_class not in final_sub_class_confidence:
                final_sub_class_confidence[sub_class] = combined_sub_class_feature_confidence[feature]
            else:
                final_sub_class_confidence[sub_class] = max(final_sub_class_confidence[sub_class], combined_sub_class_feature_confidence[feature])
        confidence = torch.tensor(list(final_sub_class_confidence.values()))
        probs = torch.nn.functional.softmax(confidence, dim=0)
        # print(probs)
        final_probs = {key: prob.item() for key, prob in zip(final_sub_class_confidence.keys(), probs)}
        sorted_final_probs = dict(sorted(final_probs.items(), key=lambda x: x[1], reverse=True))
        final_preds = []
        cumulative_prob = 0
        for sub_class, prob in sorted_final_probs.items():
            cumulative_prob += prob
            final_preds.append(sub_class)
            if cumulative_prob >= self.p_threshold:
                break
        final_preds = set(final_preds)
        alive = False
        if cifar_class.lower() in living:
            alive = True
        if alive:
            for i in remove_dict['living']:
                final_preds.discard(i)
        else:
            for i in remove_dict['non-living']:
                final_preds.discard(i)
        final_preds = list(final_preds)
        final_preds_extended = final_preds.copy()
        for i in final_preds:
            if i in tree_dict:
                
                final_preds_extended.extend(tree_dict[i])
        final_preds_extended = list(set(final_preds_extended))
        return final_preds_extended

    def predict(self, image_path):
        cifar_class = self.cifar_class(image_path)
        broad_class_preds = self.classify_broad(image_path, cifar_class)
        sub_class_probs_initial = self.sub_class_probs_initial(image_path, broad_class_preds)
        sub_class_feature_probs = self.sub_class_feature_probs(image_path, broad_class_preds)
        combined_sub_class_feature_confidence = self.combine_probs(broad_class_preds, sub_class_probs_initial, sub_class_feature_probs)
        final_preds = self.final_predictions(cifar_class, combined_sub_class_feature_confidence)
        self.results.loc[len(self.results)] = [image_path, final_preds]
        return final_preds

    def generate_descriptions(self, image_path, final_preds):
        explaination_prompt = f"This image of a/an {self.cifar_class(image_path)} is AI generated it has {[[i] for i in final_preds]} as artifacts.  Explain how it is present in the image"
        explainations = self.vlm_prediction(explaination_prompt, image_path)

        return explainations


##for creating heatmaps which help in explainability in SigLIP
    def generate_heatmap(self,image_path):
      patch_size = (6,6)
      image = Image.open(image_path)
      image_array = np.array(image)
      
      h,w,c= image_array.shape
      ph,pw = patch_size



      heatmap = np.zeros((h,w))

      cifar_class = self.cifar_class(image_path)
      broad_class_preds = self.classify_broad(image_path, cifar_class)
      sub_class_probs_initial = self.sub_class_probs_initial(image_path, broad_class_preds)
      sub_class_feature_probs = self.sub_class_feature_probs(image_path, broad_class_preds)
      combined_sub_class_feature_confidence = self.combine_probs(broad_class_preds, sub_class_probs_initial, sub_class_feature_probs)

      for i in range(0,h-ph+1,3):
        for j in range(0,w-pw+1,3):
          
          mimg = image_array.copy()
          mimg[i:i+ph,j:j+pw,:] = 0

          if mimg.min() < 0 or mimg.max() > 255:
              mimg = (255 * (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))).astype(np.uint8)
          elif mimg.dtype != np.uint8:
              mimg = mimg.astype(np.uint8)


          image1 = Image.fromarray(mimg)

          image1.save("/content/1.png")



          cifar_class1 = cifar_class
          broad_class_preds1 = broad_class_preds
          sub_class_probs_initial1 = self.sub_class_probs_initial("/content/1.png", broad_class_preds1)
          sub_class_feature_probs1 = self.sub_class_feature_probs("/content/1.png", broad_class_preds1)
          combined_sub_class_feature_confidence1 = self.combine_probs(broad_class_preds1, sub_class_probs_initial1, sub_class_feature_probs1)
          cosineval =0
          c =0
          for keys in combined_sub_class_feature_confidence:
            c+=1
            cosineval += abs(combined_sub_class_feature_confidence[keys]- combined_sub_class_feature_confidence1[keys])
          heatmap[i:i+ph,j:j+pw] += (cosineval/c)


      plt.figure(figsize =(10,10))
      plt.imshow(heatmap,cmap = "hot",interpolation = "nearest")
      plt.colorbar(label="difference")
      plt.title("heatmap")
      plt.xlabel("Patch X-coordinate")
      plt.ylabel("Patch Y-coordinate")
      plt.show()


##enter your model names / paths wherever applicable
sota_classifier = ""
siglip_model = ""
qwen_path = ""
pipe = task2_pipeline(device, sota_classifier, siglip_model, qwen_path , artifact_classification, class_desc_sub, sub_class_features_dict, feature_to_class)

image_path = ""  ##put your image path
preds =pipe.predict(image_path)
explanation = pipe.generate_descriptions(image_path,preds)

print(preds)
print(explanation)

