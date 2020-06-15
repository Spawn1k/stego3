import numpy as np
import cv2
import sys

from math import log10, sqrt
from PIL import Image

quantization_table = np.array([[16,11,10,16,24,40,51,61],      
                    [12,12,14,19,26,58,60,55],    
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
					
def dct2(block):
    return np.round(cv2.dct(block.astype(np.float32) - 128) / quantization_table)

def idct2(block):
    return np.round(cv2.idct(block * quantization_table) + 128)

def PSNR(image,secret_image):
    image = cv2.imread(image)
    secret_image = cv2.imread(secret_image)
    MSE = np.mean((image - secret_image) ** 2)
    if MSE == 0:
        print('PSNR: infinity\n')
    else:
        PSNR = 10 * log10(255 ** 2 / MSE)
        print('PSNR:', PSNR, '\nRMSE:', sqrt(MSE), '\n')

def hide(image,message):
    row = 4
    col = 3
    try:
        message = message.encode('ascii')
    except UnicodeEncodeError:
        print(f'Ascii can\'t encode "{message}"')
        sys.exit(1)
    message = ''.join(format(i, '0>7b') for i in message) + '0' * 8
    image = cv2.imread(image)
    if message == '00000000':
        print('No message')
        cv2.imwrite('secret.bmp', image)
        return None
    if (image.size / 3 // 64) < len(message):
        print('Too large messsage')
        sys.exit(1)
    secret_image = image.copy()
    pointer = 0
    for i in range(0, secret_image.shape[0], 8):
        for j in range(0, secret_image.shape[1], 8):
            block = dct2(secret_image[i:i+8,j:j+8,2])
            block[row, col] +=  - block[row, col] % 2 + int(message[pointer])
            #print(block[row, col]) #debug
            secret_image[i:i+8,j:j+8,2] = idct2(block)
            pointer += 1
            if pointer == len(message):
                break
        if pointer == len(message):
            break   
    cv2.imwrite('secret.bmp', secret_image)

def extract(image):
    row = 4
    col = 3
    img = cv2.imread(image)[:,:,2]
    message = ''
    text = ''
    end = '0' * 8
    for i in range(0,img.shape[0],8):
        for j in range(0, img.shape[1], 8):
            block = dct2(img[i:i+8, j:j+8])
            message += str(int(block[row, col]) % 2)
            if message[-8:] == end:
                break
        if message[-8:] == end:
                break    
    if message[-8:] != end or len(message) == 8:
        print('Message was not found or it is NULL')
        return ''
    #print(message) #debug
    text = ''.join([chr(int(message[i:i+7], 2)) for i in range(0, len(message) - 8, 7)])
    return text

inpimage = 'plato.bmp'
outpimage = 'secret.bmp'
opt = '1'
while opt in ['1', '2', '3']:
    opt = input('Input 1 to hide and input 2 to extract,\ninput 3 for PSNR only: ')
    if opt == '1': 
        hide(inpimage, input('Message to be hidden:\n'))
    elif opt == '2': 
        print(extract(outpimage), '\n')
    elif opt == '3':
        PSNR(inpimage,outpimage)