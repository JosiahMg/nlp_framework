from dateset import apps_reviews
from models.bert_pretrained import SentimentClassifierAppsReviews
from sklearn.metrics import confusion_matrix, classification_report

dataloader = apps_reviews.get_apps_reviews()

trainer = SentimentClassifierAppsReviews(dataloader)

trainer.train()

model = trainer.get_model()
y_review_texts, y_pred, y_pred_probs, y_test = trainer.get_predictions(model, dataloader['test'])
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

