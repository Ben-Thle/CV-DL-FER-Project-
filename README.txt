Facial Expression Recognition (FER) Project

CV-DL Codebase by: Petros Mavridis, Vinzenz Obermaier, Ben Thiele, Theodoros Tsimpoukis.

Install the dependencies inside requirements.txt, eg.: pip install -r requirements.txt or each one manually.
Make sure you are using python version 3.11.
The best way to download our dataset is though this link: https://www.kaggle.com/datasets/msambare/fer2013, altough it requires creating an account, it is the best availible source for FER2013.
The download will be a .zip file, to preprocess the data for our training, the .zip file needs to be unpacked and moved to \CV-DL-FER-Project-\src\datasets, the name needs to be "FER2013". 
You can now run the \CV-DL-FER-Project-\src\pipeline\pipeline.py script to fully prepare the data four our training model. Make sure the data is is correctly split into each folder.
You can now run any training /evaluation script.

For the Folder/ Video and Cam Demo and Training, as well as the Inference.py, you need to start the program from the command shell.
Navigate to \CV-DL-FER-Project, and then execute the commands "python -m src.camDemo.Demo", "python -m src.camDemo.FolderDemo", "python -m src.camDemo.VideoDemo", 
"python -m src.camDemo.Inference", "python -m src.training.trainingFinalFinal", to execute the corresponding scripts.