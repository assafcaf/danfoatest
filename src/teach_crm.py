from experiment_runner import CRMRunner
import sys

if __name__ == '__main__':
    args = CRMRunner.parse_args()
    runner = CRMRunner(args.config)
    runner.run()
    exit()
