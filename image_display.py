def display_image(image, label, time_to_display):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import time
    #plt.rcParams['figure.dpi'] = 300
    #plt.rcParams['savefig.dpi'] = 300
    plt.imshow(mpimg.imread(image), cmap='gray')
    plt.title(label)
    plt.show(block=False)
    plt.pause(time_to_display)
    plt.close()

def main():
    display_image('./PrivateTest_1221822.jpg', 'angry', 3)

if __name__ == "__main__":
    main()
