import sys

def main():
    if len(sys.argv) == 1:
        print("No arguments provided. -h for help")
        return
    elif sys.argv[1] == "-h":
        print("Help: \n -h: Help \n -v: Version \n A program to calculate the quality of a Papplu hand. \n Provide the 21 cards as arguments ")
        return
    print("Hello, world!")




if __name__ == '__main__':
    main()