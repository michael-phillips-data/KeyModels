The actual parameters used for training are in /info
192_kazutsugi_optimization-and-validation-results.pkl and
apprentice_key_optimization-and-validation-results.pkl

Use pandas.read_pickle() as the unpickler

unpickling the models probably won't work because of folder structure. 
It will require retraining if you want to actually run the models.

There is no reason for any parameter choices other than "A lot of cross-validation experiments selected them as highest Sharpe combination over training data"
