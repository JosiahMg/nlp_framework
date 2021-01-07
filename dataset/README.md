# nlp_framework
## apps_reviews

- filename:  
  apps_reviews.csv
- format:

|content|score|
|-------|-----|
- used:

|content(x_data)|score(y_target)|
|---------------|---------------|

- dataset

|review_text|input_ids|attention_mask|targets|
|-----------|---------|--------------|-------|
|raw text of x_data|encoding of x_data|attention|y_data|

- dataloader

|train|test|validation|
|-----|----|----------|
>from nlp_framework.preprocess import apps_reviews  
>dataloader = apps_reviews.get_apps_reviews()



