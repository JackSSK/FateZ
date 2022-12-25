from Bio import motifs
import fatez.tool.sequence as seq
from pkg_resources import resource_filename
import fatez.tool.transfac as transfac


def motif_scan(peak_annotations, specie: str = 'mouse', ref_seq_path: str = 'a'):
    """
    This function calculate binding score for all motifs
    in specific regions (overlap region between peak and
    gene promoter).
    """
    peak_list = list(peak_annotations.keys)
    ### load tf motif relationships
    path = resource_filename(
        __name__, '../data/' + specie + '/Transfac201803_MotifTFsF.txt.gz'
    )
    ### make TFs motifs dict
    tf_motifs = transfac.Reader(path=path).get_motifs()
    motifs_use = [tf_motifs.keys()]

    # motif_db = pd.read_table(path)
    # TF_motif_dict = {}
    # for i in motif_db.index:
    #     TFs = motif_db.iloc[i, :][3]
    #     TF_list = TFs.split(';')
    #     Motif = motif_db.iloc[i, :][0]
    #     for i  in TF_list:
    #         if i in TF_motif_dict.keys():
    #             TF_motif_dict[i].append(Motif)
    #         else:
    #             TF_motif_dict[i] = [Motif]

    ### load reference seq
    ref_seq = seq.Reader(path=ref_seq_path).get_fasta()

    ### load TRANSFAC PWM
    pwm_path = resource_filename(
        __name__, '../data/' + specie + '/Transfac_PWM.txt'
    )
    handle = open(pwm_path)
    record = motifs.parse(handle, "TRANSFAC")
    handle.close()
    score_all = 0
    peak_motif_dict = {}
    ### motif discovering
    for peak in peak_list:
        if peak_annotations[peak] != None:
            peak2 = peak.split(':')
            ### extract peak sequence
            chr = peak2[0]
            start = int(peak2.split('-')[0])
            end = int(peak2.split('-')[1])
            peak_sequence = ref_seq[chr][start - 1:end]

            motif_score_dict = {}
            ### motif discovering
            for i in motifs_use:

                motif_use_name = i[1:5]
                motif = record[int(motif_use_name)]
                pwm = motif.counts.normalize()
                pssm = pwm.log_odds()
                for position, score in pssm.search(peak_sequence, threshold=0):
                    motif_score_dict[i] = score
            peak_annotations.keys[peak] = motif_score_dict