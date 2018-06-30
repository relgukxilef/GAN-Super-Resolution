import argparse
import tensorflow as tf

from model import GANSuperResolution

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--i', type=str, default='', help='input file')

args = parser.parse_args()

def main(_):
    tf.reset_default_graph()

    with tf.Session() as session:
        model = GANSuperResolution(session)
        model.train() if args.phase == 'train' else model.scale_file(args.i)

if __name__ == '__main__':
    tf.app.run()
