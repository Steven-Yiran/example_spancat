"""Bert for span classification"""
from typing import Optional, Tuple, Union

import torch
import json
import os
from torch.nn import BCEWithLogitsLoss

from transformers import (
    BertForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

class BertForSpanClassification(BertForTokenClassification):
    """
    Bert model for span classification. Span classification can be modelled as a multi-label token classification task.
    """
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QA:   
    """
    Holds a span classification model for question answering.
    """ 
    def __init__(self, model_path: str):
        self.model = BertForSpanClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if os.path.exists(f"{model_path}/label_map.json"):
            self.label_map = json.load(open(f"{model_path}/label_map.json"))
        else:
            print("No label map found, please train the model with the QATrainer class")


    def predict(self, example):
        """
        Predict the answer span for a given question and context
        """
        question = example["question"]
        context = example["context"]
        answer = example["answers"]

        raw_encoded_example = self.tokenizer(question, context, return_offsets_mapping=True, return_tensors="pt")
        tokens = self.tokenizer.batch_decode(raw_encoded_example['input_ids'].tolist()[0])

        #encoded_example = tokenizer(question, context, return_tensors="pt")
        offset = raw_encoded_example.pop("offset_mapping")[0]
        with torch.no_grad():
            logits = self.model(**raw_encoded_example)["logits"][0]
        pred_probs = torch.sigmoid(logits).detach().numpy()

        label = []
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = raw_encoded_example.sequence_ids(0)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is [0] * len(input)
        label = [[0 for _ in self.label_map.keys()] for _ in range(len(offset))]

        if offset[context_start][0] <= start_char and offset[context_end][1] >= end_char:
            # Otherwise the tokens inside the answer span are labeled with 1
            answer_start = context_start
            while answer_start <= context_end and offset[answer_start][0] <= start_char:
                answer_start += 1
            answer_start = answer_start - 1

            answer_end = context_end
            while idx >= context_start and offset[answer_end][1] >= end_char:
                answer_end -= 1
            answer_end = answer_end + 1
            
            # mark label[answer_start:answer_end+1] as 1
            for j in range(answer_start, answer_end + 1):
                label[j][self.label_map['ANS']] = 1

        return pred_probs, tokens, label
 
    

class QATrainer:
    """
    Use this class to train a span classification model for question answering
    """
    def __init__(self, qa_model, max_seq_length, label_map):
        self.model = qa_model.model
        self.tokenizer = qa_model.tokenizer
        self.max_seq_length = max_seq_length
        self.label_map = label_map

    def train(
            self,
            dataset,
            output_dir: str,
            num_train_epochs: int
        ):
        train_ds = dataset.map(
            self._preprocess_training_examples,
            batched=True,
            remove_columns=dataset.column_names)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
        )
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_ds,
        )

        trainer.train()
        trainer.save_model()

        # also save the labelmap
        with open(f"{output_dir}/label_map.json", "w") as f:
            json.dump(self.label_map, f)


    def _preprocess_training_examples(self, examples):
        """
        Preprocess the training examples and turn offset span labels into token-level labels
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            return_offsets_mapping=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        labels = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is [0] * len(input)
            label = [[0 for _ in self.label_map.keys()] for _ in range(self.max_seq_length)]

            if offset[context_start][0] <= start_char and offset[context_end][1] >= end_char:
                # Otherwise the tokens inside the answer span are labeled with 1
                answer_start = context_start
                while answer_start <= context_end and offset[answer_start][0] <= start_char:
                    answer_start += 1
                answer_start = answer_start - 1

                answer_end = context_end
                while idx >= context_start and offset[answer_end][1] >= end_char:
                    answer_end -= 1
                answer_end = answer_end + 1
                
                # mark label[answer_start:answer_end+1] as 1
                for j in range(answer_start, answer_end + 1):
                    label[j][self.label_map['ANS']] = 1
            
            labels.append(label)

        inputs["labels"] = labels
        return inputs