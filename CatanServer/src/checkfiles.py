import firstNN as firstNNBot
import secondNN as secondNNBot
import thirdNN as thirdNNBot
import fourthNN as fourthNNBot
import os
import torch

if __name__ == "__main__":
    folder = input("Enter the folder name: ")
    
    # loop through all the files in the folder
    for file in os.listdir(folder):
        if not file.endswith(".pth"):
            continue
        
        print("Loading file: " + file)
        
        try:
            # load the model
            bot = firstNNBot.firstNN()
            bot.model.load_state_dict(torch.load(folder + "/" + file))
            
            # rename the file
            os.rename(folder + "/" + file, folder + "/firstNN_" + file)
            continue
        except Exception as e:
            print("Failed to load firstNN model")
        
        try:
            # load the model
            bot = secondNNBot.secondNN()
            bot.model.load_state_dict(torch.load(folder + "/" + file))
            
            # rename the file
            os.rename(folder + "/" + file, folder + "/secondNN_" + file)
            continue
        
        except Exception as e:
            print("Failed to load secondNN model")
        
        try:
            # load the model
            bot = thirdNNBot.thirdNN()
            bot.model.load_state_dict(torch.load(folder + "/" + file))
            
            # rename the file
            os.rename(folder + "/" + file, folder + "/thirdNN_" + file)
            continue
        except Exception as e:
            print("Failed to load thirdNN model")
        
        try:
            # load the model
            bot = fourthNNBot.forthNN()
            bot.model.load_state_dict(torch.load(folder + "/" + file))
            
            # rename the file
            os.rename(folder + "/" + file, folder + "/fourthNN_" + file)
            continue
        except Exception as e:
            print("Failed to load fourthNN model")
    
    