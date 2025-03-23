from experiment_runner import PRMRunner

if __name__ == '__main__':
    args = PRMRunner.parse_args()
    runner = PRMRunner(args.config)
    runner.run()
    exit()
