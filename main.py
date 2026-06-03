from pubmed import parse_pubmed_xml, build_dataset

def main():
    print(build_dataset("data/pubmed26n1330.xml.gz"))


if __name__ == "__main__":
    main()
