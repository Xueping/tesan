import pandas as pd
import re


class ICD_Ontology():
    
    def __init__(self,icd_file, dx_flag):
        self.icd_file = icd_file
        self.dx_flag = dx_flag
        self.rootLevel()


    def rootLevel(self):

        df = pd.read_csv(self.icd_file,index_col=0, dtype=object)

        dxs = df.ICD9_CODE.tolist()
        dxMaps = dict()
        
        if self.dx_flag:
            
            for dx in dxs:
                dxMaps.setdefault(dx[0:3], 0)

            for k in dxMaps.keys():
                if k >= '001' and k <= '139':
                    dxMaps[k] = 1
                if k >= '140' and k <= '239':
                    dxMaps[k] = 2
                if k >= '240' and k <= '279':
                    dxMaps[k] = 3
                if k >= '280' and k <= '289':
                    dxMaps[k] = 4
                if k >= '290' and k <= '319':
                    dxMaps[k] = 5
                if k >= '320' and k <= '389':
                    dxMaps[k] = 6
                if k >= '390' and k <= '459':
                    dxMaps[k] = 7
                if k >= '460' and k <= '519':
                    dxMaps[k] = 8
                if k >= '520' and k <= '579':
                    dxMaps[k] = 9
                if k >= '580' and k <= '629':
                    dxMaps[k] = 10
                if k >= '630' and k <= '679':
                    dxMaps[k] = 11
                if k >= '680' and k <= '709':
                    dxMaps[k] = 12
                if k >= '710' and k <= '739':
                    dxMaps[k] = 13
                if k >= '740' and k <= '759':
                    dxMaps[k] = 14
                if k >= '760' and k <= '779':
                    dxMaps[k] = 15
                if k >= '780' and k <= '799':
                    dxMaps[k] = 16
                if k >= '800' and k <= '999':
                    dxMaps[k] = 17
                if k >= 'E00' and k <= 'E99':
                    dxMaps[k] = 18
                if k >= 'V01' and k <= 'V90':
                    dxMaps[k] = 19
            self.rootMaps = dxMaps
            
        else:
            
            for dx in dxs:
                dxMaps.setdefault(dx[0:2], 0)

            for k in dxMaps.keys():
                if k == '00':
                    dxMaps[k] = 1
                if k >= '01' and k <= '05':
                    dxMaps[k] = 2
                if k >= '06' and k <= '07':
                    dxMaps[k] = 3
                if k >= '08' and k <= '16':
                    dxMaps[k] = 4
                if k >= '17':
                    dxMaps[k] = 5
                if k >= '18' and k <= '20':
                    dxMaps[k] = 6
                if k >= '21' and k <= '29':
                    dxMaps[k] = 7
                if k >= '30' and k <= '34':
                    dxMaps[k] = 8
                if k >= '35' and k <= '39':
                    dxMaps[k] = 9
                if k >= '40' and k <= '41':
                    dxMaps[k] = 10
                if k >= '42' and k <= '54':
                    dxMaps[k] = 11
                if k >= '55' and k <= '59':
                    dxMaps[k] = 12
                if k >= '60' and k <= '64':
                    dxMaps[k] = 13
                if k >= '65' and k <= '71':
                    dxMaps[k] = 14
                if k >= '72' and k <= '75':
                    dxMaps[k] = 15
                if k >= '76' and k <= '84':
                    dxMaps[k] = 16
                if k >= '85' and k <= '86':
                    dxMaps[k] = 17
                if k >= '87' and k <= '99':
                    dxMaps[k] = 18
            dxMaps['E'] = 19
            dxMaps['V'] = 20
            self.rootMaps = dxMaps

    def getRootLevel(self,code):
        
        if self.dx_flag:
            root = code[0:3]
        else:
            if code.startswith('E'):
                root = 'E'
            elif code.startswith('V'):
                root = 'E'
            else:
                root = code[0:2]
        return self.rootMaps[root]
    
class CCS_Ontology():
    
    def __init__(self,ccs_file):
        self.ccs_file = ccs_file
        self.rootLevel()
        
    def rootLevel(self):

#         ccs_file = '../data/CCS/SingleDX-edit.txt'
        with open(self.ccs_file) as f:
            content = f.readlines()

        pattern_code = '^\w+' #match code line in file
        pattern_newline = '^\n'#match new line '\n'

        prog_code = re.compile(pattern_code)
        prog_newline = re.compile(pattern_newline)

        catIndex = 0
        catMap = dict() # store index:code list
        codeList = list()
        for line in content:
            
            #if the current line is code line, parse codes to a list and add to existing code list.
            result_code = prog_code.match(line)
            if result_code:
                codes = line.split()
                codeList.extend(codes)

            #if current line is a new line, add new index and corresponding code list to the catMap dict.
            result_newline = prog_newline.match(line)
            if result_newline:
                catMap[catIndex] = codeList
                codeList = list() # initualize the code list to empty
                catIndex += 1 #next index
                
        code2CatMap = dict()
        for key, value in catMap.items():
            for code in value:
                code2CatMap.setdefault(code, key)

        self.rootMaps = code2CatMap
    
    def getRootLevel(self,code):
        return self.rootMaps[code]
