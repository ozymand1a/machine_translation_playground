import spacy

import torch


class Translator:
    def __init__(
            self,
            model,
            device
    ):
        self.model = model
        self.model.eval()

        self.device = device

    def translate_sentence(
            self,
            sentence,
            src_field,
            trg_field,
            max_len=50
    ):
        if isinstance(sentence, str):
            nlp = spacy.load('de')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(self.device)

        src_len = torch.LongTensor([len(src_indexes)])

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor, src_len)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):

            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)

            with torch.no_grad():
                output, hidden = self.model.decoder(trg_tensor, hidden, encoder_outputs, mask)

            pred_token = output.argmax(1).item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:]

    def translate_sentence_with_attention(
            self,
            sentence,
            src_field,
            trg_field,
            max_len=50
    ):
        if isinstance(sentence, str):
            nlp = spacy.load('de')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(self.device)

        src_len = torch.LongTensor([len(src_indexes)])

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor, src_len)

        mask = self.model.create_mask(src_tensor)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        attentions = torch.zeros(max_len, 1, len(src_indexes)).to(self.device)

        for i in range(max_len):

            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)

            with torch.no_grad():
                output, hidden, attention = self.model.decoder(trg_tensor, hidden, encoder_outputs, mask)

            attentions[i] = attention

            pred_token = output.argmax(1).item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
