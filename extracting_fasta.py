import urllib.request
#import urllib
import csv
from statistics import mean
import math
from matplotlib import pyplot as plt
import numpy as np
import time
import argparse

def write_outputfile(unq_pro_num, homodimers, avg_len, max_len, min_len):
    print("write outputfile")
    trimester = time.strftime("_%Y_%m_%d-%H:%M:%S")
    filename = "output" + trimester + ".txt"
    file1 = open(filename, "w")
    str1 = ["The num of unique proteins is: " + str(unq_pro_num) +" \n",
           "The num of homodimers is: " + str(len(homodimers)) + " \n",
           "Avg len of protein sequences are: "+ str(round(avg_len)) +"\n",
           "Max leng of protein sequences are: "+ str(max_len) +"\n",
           "Min leng of protein sequences are: "+ str(min_len) + "\n",
           "\n" + "list of Homodimers" + " \n"]

    # \n is placed to indicate EOL (End of Line)
    file1.writelines(str1)
    for x in range(len(homodimers)):
        file1.write(str(homodimers[x])+ "\n")
    file1.close()  # to change file access modes

def draw_histogram(length_distribution):
    print("draw histogram")
    avg_len = mean(length_distribution)
    max_len = max(length_distribution)
    min_len = min(length_distribution)
    bin_interval = (max_len - min_len)/round(math.sqrt(len(length_distribution)))
    data = np.array(length_distribution)
    fig, axs = plt.subplots(figsize=(10, 7))
    axs.hist(data, bins = round(math.sqrt(len(length_distribution))))
    plt.title('Protein sequence length distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    return (avg_len, max_len, min_len)


"""Prints out the headers and sequences in a final .csv file"""
def write_file( header_A, sequencs_A, header_B, sequencs_B, final_file, label):
    print("Write file")
    filename1 = "/home/at/work/ECEN_404/extracted_data/"
    trimester = time.strftime("_%Y_%m_%d-%H:%M:%S")
    filename = filename1 + final_file + trimester +".csv"
    length_distribution = []
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        for x in range(len(header_A)):
            row1 = [header_A[x], sequencs_A[x], header_B[x], sequencs_B[x],label]
            csvwriter.writerow(row1)
            length_distribution.append(len(sequencs_A[x]))
            length_distribution.append(len(sequencs_B[x]))
    return (length_distribution)


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
            split_data_A = str(A).split('\\n')
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
            #link_B = urllib.urlopen(B)
            #link_A = urllib.urlopen(A)

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
    homodimers = []
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
                if(protienA_id == protienB_id):
                    homodimers.append([protienA_id, protienB_id])

            except:
                print("Error to make link: ", protienA_id, protienB_id)

    pro_ID = list(set(proA_links + proB_links))
    return proA_links, proB_links, len(pro_ID), homodimers


"""opens and reads a txt or csv file"""
def open_file(filename, csv, low_lmt, hgh_lmt):
    if csv == 0:
        #text_file = open(filename, "r")
        #lines = text_file.readlines()
        final_file = "iRefWeb_(" + str(low_lmt) +"_"+ str(hgh_lmt)+")"
        lines = []
        with open(filename, "r") as f:
            i = 0
            for line in f:
                if (i >= low_lmt):
                    lines.append(line)
                i += 1
                if (i == hgh_lmt):
                    break
        proA_links, proB_link, unq_pro_num, homodimers = fasta_link(lines, csv)
        data_A, data_B = extract_sequence(proA_links, proB_link)
        header_A, sequencs_A = divide_data(data_A)
        header_B, sequencs_B = divide_data(data_B)
        length_distribution = write_file(header_A, sequencs_A, header_B, sequencs_B, final_file, label = 1)
        avg_len, max_len, min_len = draw_histogram(length_distribution)
        write_outputfile(unq_pro_num, homodimers, avg_len, max_len, min_len)

    else:
        lines = []
        with open (filename, "r") as f:
            i = 0
            for line in f:
                if (i >= low_lmt):
                    lines.append(line)
                i += 1
                if (i == hgh_lmt):
                    break
        proA_links, proB_link, unq_pro_num, homodimers = fasta_link(lines, csv)
        data_A, data_B = extract_sequence(proA_links, proB_link)
        header_A, sequencs_A = divide_data(data_A)
        header_B, sequencs_B = divide_data(data_B)
        length_distribution = write_file(header_A, sequencs_A, header_B, sequencs_B, final_file ="negatome", label = 0)
        avg_len, max_len, min_len = draw_histogram(length_distribution)
        write_outputfile(unq_pro_num, homodimers, avg_len, max_len, min_len)

def main():
    """
    parser = argparse.ArgumentParser(description= 'Required parameters to run the script')
    parser.add_argument('-ll','--low_lmt',type=int, help ='first line to start fasta extraction')
    parser.add_argument('-hl','--hgh_lmt',type=int, help ='last line to end fasta extraction')
    args = parser.parse_args()
    """



    open_file(filename="/home/at/work/ECEN_404/negatome/combined.csv", csv = 1, low_lmt=0, hgh_lmt=6532)
    print("DONE")


if __name__ == "__main__":
    main()
