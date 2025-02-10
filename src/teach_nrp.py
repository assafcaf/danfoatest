from experiment_runner import NRPRunner
if __name__ == '__main__':
    args = NRPRunner.parse_args()
    runner = NRPRunner(args.config)
    runner.run()
    
