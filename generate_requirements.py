import subprocess


def generate_requirements():

    # Run 'pip freeze' and capture the output
    result = subprocess.run(['pipreqs', './src', '--print'], stdout=subprocess.PIPE, text=True)
    # Get the output as a list of strings (each string represents a package)
    packages = result.stdout.splitlines()

    # Run 'pip freeze' and capture the output
    result = subprocess.run(['pipreqs', './app', '--print'], stdout=subprocess.PIPE, text=True)
    # Get the output as a list of strings (each string represents a package)
    packages.extend(result.stdout.splitlines())

    # Run 'pip freeze' and capture the output
    result = subprocess.run(['pipreqs', '--scan-notebooks', './notebooks', '--print'], stdout=subprocess.PIPE, text=True)
    # Get the output as a list of strings (each string represents a package)
    packages.extend(result.stdout.splitlines())

    ignore_packages = ['db_utils', 'predict']
    # Write the list of packages to the 'requirements.txt' file
    with open('requirements.txt', 'w') as f:
        for package in set(packages):
            if all([ig not in package for ig in ignore_packages]):
               f.write(package + '\n')


if __name__ == "__main__":
    generate_requirements()
