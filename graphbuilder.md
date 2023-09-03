## Docs for Graph Construction Code (BA Henschke)

Uses data from BGP Route Collectors to create an AS-level graph of the Internet topology, and enriches it with various node- and link-attributes.

### Requirements:
- version of python that preserves dict order (3.7+)
- [BGPStream v2.2](https://bgpstream.caida.org/docs/install/pybgpstream)
- [pytricia](https://github.com/jsommers/pytricia)
- [graph_tool](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#gnulinux)

### Usage:
- Obtain a PeeringDB json file from https://publicdata.caida.org/datasets/peeringdb/ (daily release)
- Obtain an AS2org json file from https://publicdata.caida.org/datasets/as-organizations/ (quarterly release)
- If the above files are omitted, no IXPs will be filtered from paths, and all organization attributes will be NA
- `python3 GraphBuilder.py -h` for help on options
- e.g. `python3 GraphBuilder.py -p peeringdb_2022_03_01.json -o as2org_2022_04_01.jsonl -s '2022-04-01 07:50:00' -e '2022-04-01 08:10:00'`

### Relationship Inference:
- Follow instructions at https://github.com/Zitong-Jin/TopoScope#basic-inference to infer ASRank and TopoScope inferences
- If ASRank inferences with a manually input clique are desired, run asrank.pl with the `--clique` option
- `relationships.py` provides functions to 
    - add relationship inferences to a graph_tool graph 
    - turn a file containing relationship inferences into a graph_tool graph
- Code in `TopoScope/uniquePath.py` and `asrank.pl` has been changed to match the bogon filtering in `GraphBuilder.py`
- Code in `asrank.pl` has been changed to work with python3 (originally used python2 print statement)
- Many more potential features are described in the ProbLink and TopoScope papers, and code to calculate these features can 
be found in the their respective github repos

### BGPStream Notes:
- Don't set start and/or end time on the exact time of the RIB dump. Leave some time before and after, e.g., 07:50:00 - 08:10:00 instead of something like 08:00:00 - 08:10:00
- Newly released route collectors are not frequently added to the BGPStream data broker. One should occasionally check if the [broker](https://bgpstream.caida.org/data) is up to date 
- The cache- and RIB period filter- functionality of BGPstream should not be used at the same time (see [here](https://github.com/CAIDA/libbgpstream/issues/223))
- There are very frequent `'HTTP ERROR: Failure when receiving data from the peer (56)'` warnings, which can be safely ignored

### Edge Cases to keep in mind:
- My scripts do not filter AS confederate sequences, but their occurence is exceedingly rare (encountered only once in 2009-01-15) - will crash on encounter
- In exceedingly rare cases, route collectors dump an empty path default route (pfx 0.0.0.0 and empty AS path)
