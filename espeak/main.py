import os

class SpeechGenerator:
    def __init__(self, fromFilePath=None, savespeech=False):
        self.file = fromFilePath
        self.savespeech = savespeech

    def speak(self, text="No text was found"):
        if self.file==None:
            if self.savespeech==False:
                if type(text)!=str:
                    raise ValueError()
                    print("The text given was not understood")
                Command = "bash glados.sh"+" "+"\""+text+"\""
                os.system(Command)
            else:
                Command = "bash glados.sh"+" "+"\""+text+"\""
                if type(Command)!=str:
                    raise ValueError()
                    print("The text given was not understood")
                os.system(Command)
        else:
            with open(self.file, 'r') as File:
                text = File.read()
            if type(text)!=str:
                    raise ValueError()
                    print("The text given was not understood")
            Command = "bash glados.sh"+" "+"\""+text+"\""
            os.system(Command)

whatever = SpeechGenerator(fromFilePath="testdata.txt")
whatever.speak()




        

