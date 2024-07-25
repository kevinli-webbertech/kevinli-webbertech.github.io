## Keras - Model Evaluation and Model Prediction

### Model Evaluation

Model evaluation is an essential step during the development of a model to assess its performance and ensure it is well-suited for the given problem and data. Keras provides a convenient function for model evaluation called `evaluate`.

#### The `evaluate` Function

The `evaluate` function in Keras has three main arguments:

- **Test data**: The input data used to evaluate the model.
- **Test data label**: The true labels corresponding to the test data.
- **Verbose**: A boolean argument (`True` or `False`) to control the verbosity of the output.

#### Evaluating the Model

To evaluate the model, use the following code:

```python
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

output
0

The test accuracy is 98.28%.

Model Prediction

Prediction is the final step and our expected outcome of the model generation.

signature of the predict method

predict(
   x, 
   batch_size = None, 
   verbose = 0, 
   steps = None, 
   callbacks = None, 
   max_queue_size = 10, 
   workers = 1, 
   use_multiprocessing = False
)


prediction for our MPL model

pred = model.predict(x_test) 
pred = np.argmax(pred, axis = 1)[:5] 
label = np.argmax(y_test,axis = 1)[:5] 

print(pred) 
print(label)

Line 1 call the predict function using test data.

Line 2 gets the first five prediction

Line 3 gets the first five labels of the test data.

Line 5 - 6 prints the prediction and actual label.

The output of the above application is as follows −

[7 2 1 0 4] 
[7 2 1 0 4]


