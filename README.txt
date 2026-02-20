Facial Expression Recognition (FER) Project

CV-DL Codebase by: Petros Mavridis, Vinzenz Obermaier, Ben Thiele, Theodoros Tsimpoukis.

Install the dependencies inside requirements.txt, eg.: pip install -r requirements.txt or each one manually.
The best way to download our dataset is though this link: https://www.kaggle.com/datasets/msambare/fer2013, altough it requires creating an account, it is the best availible source for FER2013.
The download will be a .zip file, to preprocess the data for our training, the .zip file needs to be unpacked and moved to \CV-DL-FER-Project-\src\datasets, the name needs to be "FER2013". 
You can now run the \CV-DL-FER-Project-\src\pipeline\pipeline.py script to fully prepare the data four our training model. Make sure the data is is correctly split into each folder.
You can now run any training /evaluation script.