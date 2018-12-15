import os
import sys
import imageio
import argparse

def gendif( img_dir, output_gif_path ):
    images = []
    index = 1
    while( True ):
        img_path = os.path.join( img_dir, str(index) + '.jpg' )
        if os.path.exists( img_path ):
            images.append( imageio.imread(img_path ) )
        else:
            break;
        index += 1;
    imageio.mimsave( output_gif_path, images, duration = 1 );
    
        

if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', required = True, help = 'images dir path')
    ap.add_argument('-o', required = True, help = 'output gif path')
    args = vars(ap.parse_args())

    img_dir = args['i']
    output_gif_path = args['o']

    gendif(img_dir, output_gif_path)
