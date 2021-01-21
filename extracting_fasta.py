
import urllib.request
import csv

"""Prints out the headers and sequences in a final .csv file"""
def write_file( header_A, sequencs_A, header_B, sequencs_B, final_file):
    print("Write file")
    filename1 = "/home/at/work/ECEN_404/extracted_data/"
    filename = filename1 + final_file +".csv"
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        for x in range(len(header_A)):
            row1 = [header_A[x], header_B[x]]
            row2 = [sequencs_A[x], sequencs_B[x]]
            csvwriter.writerow(row1)
            csvwriter.writerow(row2)


"""divides the header and the sequence"""
def divide_data(data_A):
    print("derive data")
    header= []
    seqnc= []
    x = 0
    for A in (data_A):
        print(x)
        try:
            A = str(A)
            split_data_A = A.split('\\n')
            header_str = split_data_A[0]
            words = str(header_str).split('|')
            header_AA = words[1]
            split_data_A.pop(0)
            sequence_A= "".join([str(sen) for sen in split_data_A])
        except:
            print("Error found in data division: ", x)
        x = x + 1

        header.append(header_AA)
        seqnc.append(sequence_A)
    return header, seqnc

    #print("Hello from a function")



"""Goes to web following the fasta link and extract the sequence"""
def extract_sequence(proA_links, proB_link ):
    print("extract sequence")
    data_A = []
    data_B = []
    x = 0
    for A, B in zip(proA_links, proB_link):
        print(x)
        try:
            link_A = urllib.request.urlopen(A)
            link_B = urllib.request.urlopen(B)
            data_A.append(link_A.read())
            data_B.append(link_B.read())
        except:
            print("Error found to open link:", x)
        x = x + 1
    return data_A, data_B
    #print("Hello from a function")

""""reads lines and returns a fasta link from the uniprod id"""
def fasta_link(lines, csv):
    print("fasta_link")
    proA_links = []
    proB_links = []
    for line in lines:
        if csv:
            words = line.split(",")
        else:
            words = line.split()
        try:
            protienA_id = words[1]
            protienB_id = words[2].strip("\n")
        except:
            print("errror to find the ID ", words[0])
        if '_' in protienA_id or '_' in protienB_id or (protienA_id.isalpha()) or (protienB_id.isalpha()):
            continue
        else:
            try:
                protienA_link = "https://www.uniprot.org/uniprot/" + protienA_id + ".fasta"
                protienB_link = "https://www.uniprot.org/uniprot/" + protienB_id + ".fasta"
                proA_links.append(protienA_link)
                proB_links.append(protienB_link)
            except:
                print("Error to make link: ", protienA_id, protienB_id)
    return proA_links, proB_links


"""opens and reads a txt or csv file"""
def open_file(filename, csv):
    if csv == 0:
        text_file = open(filename, "r")
        lines = text_file.readlines()
        proA_links, proB_link = fasta_link(lines, csv)
        data_A, data_B = extract_sequence(proA_links, proB_link)
        header_A, sequencs_A = divide_data(data_A)
        header_B, sequencs_B = divide_data(data_B)
        write_file(header_A, sequencs_A, header_B, sequencs_B, final_file="iRefWeb_2000")
        #print("Hello from a function")
    else:
        text_file = open(filename, "r")
        lines = text_file.readlines()
        proA_links, proB_link = fasta_link(lines, csv)
        data_A, data_B = extract_sequence(proA_links, proB_link)
        header_A, sequencs_A = divide_data(data_A)
        header_B, sequencs_B = divide_data(data_B)
        write_file(header_A, sequencs_A, header_B, sequencs_B, final_file = "Negatome3")
        #print("Hello from a function")

def main():
    open_file(filename="/home/at/work/ECEN_404/iRefWeb/sample.txt", csv = 0)
    print("DONE")


if __name__ == "__main__":
    main()
