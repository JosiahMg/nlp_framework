from nlp_framework.preprocess.apps_reviews import ReviewDataloader
from nlp_framework.models.bert_pretrained import SentimentClassifierAppsReviews
from sklearn.metrics import confusion_matrix, classification_report

loadtext = ReviewDataloader()
dataloader = loadtext.get_data_loader()

trainer = SentimentClassifierAppsReviews(dataloader)

trainer.train()

model = trainer.get_model()
y_review_texts, y_pred, y_pred_probs, y_test = trainer.get_predictions(model, dataloader['test'])
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

