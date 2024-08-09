PyTorch - Loading Data

PyTorch includes a package called torchvision which is used to load and prepare the dataset. It includes two basic functions namely Dataset and DataLoader which helps in transformation and loading of dataset.

Dataset

Dataset is used to read and transform a datapoint from the given dataset

syntax

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
   download = True, transform = transform)

DataLoader is used to shuffle and batch data. It can be used to load the data in parallel with multiprocessing workers.

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
   shuffle = True, num_workers = 2)

Example: Loading CSV File

We use the Python package Panda to load the csv file. The original file has the following format: (image name, 68 landmarks - each landmark has a x, y coordinates).

landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)