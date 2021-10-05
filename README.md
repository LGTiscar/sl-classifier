# sl-classifier
## 1. Three-Sign Language word detector using LSTM Machine Learning model + Flask to run just locally

As said in the title, this is a Three-Sign Language word detector. It uses a LSTM RNN neuronal network written in Python + Keras +OpenCV + Flask as processing server. Unfortunately,
it only works in local, as WebRTC and other network communication frameworks are a bit difficult for me right now, still learning :D

## 2. How to run the app

All necessary files for the app to run are within this repository, all you have to do is:

1. Create a folder and name it as you want (SignClassifier)
2. Download the .zip with this repository files and place them inside the folder or, open CMD in windows, go to that folder using:

    `cd "PATH_TO_YOUR_FOLDER"`

    and write 

    `git clone https://github.com/LGTiscar/sl-classifier.git`

    Either way, now you have all the necessary files inside your project folder.

3. Now it is time to create a python virtual environment to run the app.py: Go to the previous CMD window and type: 

    `python (or py) -m venv "YOUR_VIRTUAL_ENVIRONMENT_NAME"`

    This will create a folder inside your project folder named like "YOUR_VIRTUAL_ENVIRONMENT_NAME". Now type:

    `cd "YOUR_VIRTUAL_ENVIRONMENT_NAME"/Scripts`, hit Enter and type:
    `Activate`


    This will
    execute the "Activate" script, so that you will be using this new Python Environment. Your CMD line should look something like this:

    `(YOUR_VIRTUAL_ENVIRONMENT_NAME) C:\Users\YOUR_USER\PATH_TO_YOUR_PROJECT_FOLDER\YOUR_VIRTUAL_ENVIRONMENT_NAME"/Scripts>`
    
    Now type `cd ..` 2 times to get back to your project main folder

4. You alredy have all the necessary files and a virtual environment running. It's time to install all the app dependencies with this command:
    
    `pip install -r requirements.txt`

5. Finally, you can run your app with this command:

    `python (or py) app.py`
    
    After some log lines, in your terminal should appear something like this:
    ```
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
     Use a production WSGI server instead.
   * Debug mode: on
   * Running on all addresses.
     WARNING: This is a development server. Do not use it in a production deployment.
   * Running on http://192.168.1.51:1024/ (Press CTRL+C to quit)
    ```
    
    This means that your local server is ip and running! Just click the `http://192.168.1.51:1024/` link and it will take you to the app. Press CTRL+C back in your terminal
    to shutdown the server and the app.
     
  
