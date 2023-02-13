import argparse
argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument('--dataset', type=str, default='reddit')
argparser.add_argument('--num-epochs', type=int, default=20)
argparser.add_argument('--num-hidden', type=int, default=16)
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--num-negs', type=int, default=1)
argparser.add_argument('--neg-share', default=False, action='store_true',
                       help="sharing neg nodes for positive nodes")
argparser.add_argument('--fan-out', type=str, default='10,25')
argparser.add_argument('--batch-size', type=int, default=10000)
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=1000)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=0,
                       help="Number of sampling processes. Use 0 for no extra process.")
args = argparser.parse_args()
print(args)
print(args.num_negs)

