import sys, json, sqlite3, argparse

class UniquePath(object):
    def __init__(self, peering_name):
        self.origin_paths = set()
        self.forward_paths = set()
        self.unique_paths = set()
        self.ixp = set()
        self.getIXP(peering_name)
        

    def getIXP(self, peeringdb_file):
        if peeringdb_file.endswith('json'):
            with open(peeringdb_file) as f:
                data = json.load(f)
            for i in data['net']['data']:
                if i['info_type'] == 'Route Server':
                    self.ixp.add(str(i['asn']))

        elif peeringdb_file.endswith('sqlite'):
            conn = sqlite3.connect(peeringdb_file)
            c = conn.cursor()
            for row in c.execute("SELECT asn, info_type FROM 'peeringdb_network'"):
                asn, info_type = row
                if info_type == 'Route Server':
                    self.ixp.add(str(asn))

        else:
            raise TypeError('PeeringDB file must be either a json file or a sqlite file.')

    # !!! MODIFIED TO BE IN LINE WITH CODE OF GRAPH CONSTRUCTION !!! - Pascal H.
    def ASNAllocated(self, asn):
        if asn == 0 or asn == 23456: # AS_TRANS
            return False
        if 64496 <= asn <= 131071: # docs and sample code, private use, reserved
            return False
        if asn >= 401309: # not assigned, private use
            return False
        return True

    def getPath(self, name):
        with open(name) as f:
            for line in f:
                if line.strip() == '':
                    continue
                self.origin_paths.add(line.strip())
                asn_list = line.strip().split('|')
                for asn in asn_list:
                    if asn in self.ixp:
                        asn_list.remove(asn)
                asn_list = [v for i, v in enumerate(asn_list)
                            if i == 0 or v != asn_list[i-1]]
                asn_set = set(asn_list)
                if len(asn_set) == 1 or not len(asn_list) == len(asn_set):
                    continue
                for asn in asn_list:
                    if not self.ASNAllocated(int(asn)):
                        break
                else:
                    self.forward_paths.add('|'.join(asn_list))
                continue

    def writePath(self):
        f = open('aspaths.txt', 'w')
        for path in self.forward_paths:
            f.write(path + '\n')
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean paths')
    parser.add_argument('-i', '--input_name', required=True)
    parser.add_argument('-p', '--peering_name', required=True)
    args = parser.parse_args()
    
    path = UniquePath(args.peering_name)
    path.getPath(args.input_name)
    path.writePath()