## Mridangam stroke classification features folder
We have set up this folder as the default location in where to store the features for the mridangam stroke classification, so that
you only need to compute these a single time. Feel free to change the location by running:

```
from compiam.timbre.stroke_classification.MridangamStrokeClassification
mridangam_stroke_classification = MridangamStrokeClassification()
mridangam_stroke_classification.computed_features_path = "/new/path/to/csv/feature/file.csv"
```