import argparse
import yaml

def generate_command_line_args(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    args = []
    for key, value in config.items():
        if not key.startswith('#'):  # Ignore commented pairs
            args.append(f'--{key} {value}')

    return ' '.join(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate command line arguments from YAML')
    parser.add_argument('yaml_file', type=str, help='Path to the YAML file')

    args = parser.parse_args()
    yaml_file = args.yaml_file

    command_line_args = generate_command_line_args(yaml_file)
    print(command_line_args)