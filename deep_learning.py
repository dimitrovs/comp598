import graphlab
from graphlab import SFrame

train_input = graphlab.image_analysis.load_images('train_images/', "auto", with_path=False, random_order=False)
train_output = SFrame.read_csv('train_outputs.csv',delimiter=',', header=True, column_type_hints=[int,int])
train_output.rename({'Prediction':'label'})
train_output.remove_column('Id')
train_output.add_column(train_input.select_column('image'),name='image')
training_data, validation_data = train_output.random_split(0.8)

training_data['image'] = graphlab.image_analysis.resize(training_data['image'], 28, 28, 1)
validation_data['image'] = graphlab.image_analysis.resize(validation_data['image'], 28, 28, 1)

mnist_net = graphlab.deeplearning.get_builtin_neuralnet('mnist')

#net = graphlab.deeplearning.create(sf, target='Prediction')

m = graphlab.neuralnet_classifier.create(training_data, target='label', network = mnist_net, validation_set=validation_data, max_iterations=200)

#test_data = graphlab.image_analysis.load_images('test_images/', "auto", with_path=False, random_order=False)

#pred = m.classify(test_data)
