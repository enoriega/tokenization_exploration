from pubmed import build_pmc_dataset

def main():
    # print(build_dataset("data/pubmed26n1330.xml.gz"))
    print(build_pmc_dataset("data/aws_pmc_sample"))


if __name__ == "__main__":
    main()
