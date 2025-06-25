import sys as sys
import argparse as argparse


def main():
    parser = argparse.ArgumentParser(description="Help Train mode", add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and quit')
    parser.add_argument('--layer', type=str, default='24 24 24', help='Number of Layer')
    parser.add_argument('--epochs', type=int, default='84', help='Number of epochs')
    parser.add_argument('--loss', type=str, default='categoricalCrossentropy', help='Methods')
    parser.add_argument('--batch_size', type=int, default='8', help='???')
    parser.add_argument('--learning_rate', type=float, default='0.0314', help='Speed of adjustments for the model')
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()

# layer, epochs, loss, batch_size, learning_rate