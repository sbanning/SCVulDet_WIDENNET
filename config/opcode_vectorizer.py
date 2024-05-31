import warnings
import numpy as np
from gensim.models import Word2Vec
from pyevmasm import disassemble_hex
import solcx
from solcx import install_solc

warnings.filterwarnings("ignore")

class OPCODE_Vectorizer:

    def __init__(self, args, filename):
        self.args = args
        self.d_path = self.args.dataset
        self.vector_length = self.args.vec_length
        self.file_path = filename

        self.PUSH_GROUP = ['PUSH1', 'PUSH2', 'PUSH3', 'PUSH4', 'PUSH5', 'PUSH6', 'PUSH7', 'PUSH8', 'PUSH9', 'PUSH10',
                      'PUSH11', 'PUSH12', 'PUSH13', 'PUSH14', 'PUSH15', 'PUSH16', 'PUSH17', 'PUSH18', 'PUSH19',
                      'PUSH20', 'PUSH21', 'PUSH22', 'PUSH23', 'PUSH24', 'PUSH25', 'PUSH26', 'PUSH27', 'PUSH28', 'PUSH29',
                      'PUSH30']

        self.SWAP_GROUP = ['SWAP1', 'SWAP2', 'SWAP3', 'SWAP4', 'SWAP5', 'SWAP6', 'SWAP7', 'SWAP8', 'SWAP9', 'SWAP10',
                      'SWAP11', 'SWAP12', 'SWAP13', 'SWAP14', 'SWAP15', 'SWAP16']

        self.DUP_GROUP = ['DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7', 'DUP8', 'DUP9', 'DUP10',
                     'DUP11', 'DUP12', 'DUP13', 'DUP14', 'DUP15', 'DUP16']

        self.LOG_GROUP = ['LOG1', 'LOG2', 'LOG3', 'LOG4']

        self.fragments = []

    def byte2opcode(self):
        opcodes = []
        bytecodes = self.getbytecode()

        for bytecode in bytecodes:
            _opcode = disassemble_hex(bytecode["bytecode"])
            opcode = self.group_opcodes(_opcode)
            row = {"opcode": opcode, "val": bytecode["val"]}
            opcodes.append(row)

        return opcodes

    def group_opcodes(self, opcodes):
        #for opcode in opcodes:
        codes = filter(None, opcodes.strip().split("\n"))
        # Process each line to extract the opcode
        clean_opcode = [line.split()[0] for line in codes]
        updated_opcodes = ['PUSH' if _opcode in self.PUSH_GROUP else _opcode for _opcode in clean_opcode]
        updated_opcodes = ['DUP' if _opcode in self.DUP_GROUP else _opcode for _opcode in updated_opcodes]
        updated_opcodes = ['SWAP' if _opcode in self.SWAP_GROUP else _opcode for _opcode in updated_opcodes]
        updated_opcodes = ['LOG' if _opcode in self.LOG_GROUP else _opcode for _opcode in updated_opcodes]

        return updated_opcodes

    def getbytecode(self):
        bytecodes = []
        solcx.install_solc("0.4.24") #version of the compiler
        install_solc(version='0.4.24')
        for code, val in self.parse_file():
            source_code = '\n'.join(code)
            print(source_code)
            cont_name = self.extract_contract_name(code)
            compiled_sol = solcx.compile_source(source_code, output_values=["bin-runtime"], solc_version="0.4.24")
            bytecode = compiled_sol['<stdin>:' + cont_name]["bin-runtime"] #extract bytecode

            row = {"bytecode": bytecode, "val": val}

            bytecodes.append(row)

        return bytecodes

    def parse_file(self):
        with open(self.file_path, "r", encoding="utf8") as file:
            fragment = []
            fragment_val = 0
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if "-" * 33 in line and fragment:
                    yield fragment, fragment_val
                    fragment = []
                elif stripped.split()[0].isdigit():
                    if fragment:
                        if stripped.isdigit():
                            fragment_val = int(stripped)
                        else:
                            fragment.append(stripped)
                else:
                    fragment.append(stripped)

    def extract_contract_name(self, contract_n):
        contract_name = ""
        for line in contract_n:
            # Split the line into words
            words = line.strip().split()
            # Look for the index of 'contract' in the words
            if 'contract' in words:
                # Find the index of 'contract' and check if there's a word after it
                index_of_contract = words.index('contract')
                if index_of_contract < len(words) - 1:
                    contract_name = words[index_of_contract + 1]

        return (contract_name)

    def install_solc(self, version):
        # Get the list of installable Solc versions
        installable_versions = solcx.get_installable_solc_versions()

        # Check if the desired version is installable
        if version not in installable_versions:
            raise ValueError(f"Solc version {version} is not installable.")

        # Install Solc
        solcx.install_solc(version)

    def add_to_fragment(self, fragment):
        self.fragments.append(fragment)

    def vectorize(self, fragment):
        vectors = np.zeros(shape=(100, 100))
        for i in range(min(len(fragment), 100)):
            vectors[i] = self.embeddings[fragment[i]]
        # for testing and verification purpose only
        # self.write_to_test_file(self.args.test_file, vectors, "final vectors after array broadcast")
        return vectors

    def write_to_test_file(self, test_file, what_to_write, custom_comment):
        f = open(test_file, "a")
        f.write(
            "\n\n\n---------------------------------- " + custom_comment + " --------------------------------------\n")
        f.write(str(what_to_write))
        f.write("\n\n\n------------------------------------------------------------------------\n\n\n")


    def train_model(self):
        # for testing and verification purpose only
        # self.write_to_test_file(self.args.test_file, self.fragments, "fragments being vectorized")
        model = Word2Vec(self.fragments, min_count=1, vector_size=100, sg=0)  # sg=0: CBOW; sg=1: Skip-Gram
        self.embeddings = model.wv #word vector

        del model
        del self.fragments
        # self.write_to_test_file(self.args.test_file, "", "embeddings after vectorization")
        # for word in self.embeddings.index_to_key:
        #     embedding_vector = self.embeddings[word]
        #     print(str(f"Word: {word}, Embedding: {embedding_vector}"))
        # return self.embeddings
