import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedPhilosopherModel(nn.Module):
    def __init__(self):
        super(EnhancedPhilosopherModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = torch.relu(self.fc1(pooled_output))
        x = self.fc2(x)
        return x


class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_len):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = str(self.questions[item])
        context = str(self.contexts[item])
        answer = self.answers[item]

        encoding = self.tokenizer.encode_plus(
            question,
            context,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'answer': torch.tensor(answer, dtype=torch.float)
        }

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        answers = d["answer"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, answers)
        correct_predictions += torch.sum(outputs == answers)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)



def train_and_save_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = EnhancedPhilosopherModel()

    # Load your dataset here
    questions = ["What is the meaning of life?", "What is philosophy?"]
    contexts = ["The meaning of life is a philosophical question.", "Philosophy is the study of general and fundamental questions."]
    answers = [42, 1]

    dataset = QADataset(questions, contexts, answers, tokenizer, max_len=128)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(data_loader) * 10  # Number of epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.MSELoss().to(device)

    for epoch in range(10):
        train_acc, train_loss = train_model(
            model,
            data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(dataset)
        )
        print(f'Epoch {epoch + 1}/{10}')
        print(f'Train loss {train_loss} accuracy {train_acc}')

    torch.save(model.state_dict(), 'enhanced_philosopher_model.pt')
    print("Model saved to enhanced_philosopher_model.pt")


def load_model():
    model = EnhancedPhilosopherModel()
    model.load_state_dict(torch.load('enhanced_philosopher_model.pt'))
    model.eval()
    return model

def answer_question(question, preprocessed_text, qa_model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
        question,
        preprocessed_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.item()