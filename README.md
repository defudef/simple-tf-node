# Simple Tensorflow for NodeJS
Ready to use algorithms to train your own models with Tensorflow for NodeJS.

## Scripts

Basic scripts:
```bash
# Install dependencies
npm install # Important: do NOT use yarn as it doesn't support npm_config variables!!!

# Build app
npm run build
```

Train model:
```bash
npm run model:train --name=NAME_OF_MODEL
```

## Example model training
1. Copy `cars.json` from `src/models/examples/cars` into `models` folder in your project root
2. Run `npm run model:train --name=example/cars`
3. Get familiar with the tabular data in the output

### Output explanation:
You should notice 6 columns:
- `(index)` - just a row number
- `Acc` - car acceleration - first **input** data
- `Weight` - car's weight in lbs - seconds **input** data
- `HP` - car's horsepower - output record from the test data
- `HP (pred)` - car's horsepower - predicted by your AI model
- `Diff` - difference between `HP` and `HP (pred)`

### It's not perfect. Should I worry?
In this case - definitely not!
First of all the model is trained only on 320 records. 80 were used to test model accuracy.
Secondly, we can see model isn't overtrained which is a good sign
Lastly, the difference between `HP` and `HP (pred)` is not that big. It's just ~12-15 on average.

Imagine you have a car, the has 2074lbs weight and accelerates to 100mph in 19 seconds. Is it possible
to predict its actual horsepower? Definitely not! In this case the model predicts it as similar as human would so we can consider this as an expected result.


## TODO:
1. Read CSV
2. Simplify code more and more
3. Add classification algorithms (SVM and KNN)
4. Add Image recognition algorithms (using CNN)
5. Add NLP algorithms (probably using RNN)
6. Add more examples
