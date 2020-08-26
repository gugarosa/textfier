import torch
from textfier.tasks import QuestionAnsweringTask

# Creates a question answering task
task = QuestionAnsweringTask(model='distilbert-base-cased-distilled-squad')

# Defines the input context
context = '''Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.'''

# Defines the input questions
questions = ['How many pretrained models are available in Transformers?',
             'What does Transformers provide?',
             'Transformers provides interoperability between which frameworks?']

# Iterate over every question
for question in questions:
    # Encodes the input
    enc_question = task.tokenizer(question, context, add_special_tokens=True, return_tensors='pt')

    # Retrieves the indexes
    idx = enc_question['input_ids'].tolist()[0]

    # Performs the question answering
    preds_start, preds_end = task.model(**enc_question)

    # Gathers the most likely beginning and ending
    answer_start = torch.argmax(preds_start)
    answer_end = torch.argmax(preds_end) + 1

    # Decodes the outputs
    decoded_answer = task.tokenizer.convert_tokens_to_string(
        task.tokenizer.convert_ids_to_tokens(idx[answer_start:answer_end]))

    print(f'Question: {question}')
    print(f'Answer: {decoded_answer}')
