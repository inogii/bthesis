### 1. Download data files

PeeringDB : [https://publicdata.caida.org/datasets/peeringdb/](https://publicdata.caida.org/datasets/peeringdb/ "https://publicdata.caida.org/datasets/peeringdb/") (daily release)
AS2org jsonl file from [https://publicdata.caida.org/datasets/as-organizations/](https://publicdata.caida.org/datasets/as-organizations/ "https://publicdata.caida.org/datasets/as-organizations/") (quarterly release)

#### 1.2 Installing requirements

-   Version of python that preserves dict order (3.7+)
```shell
$ pip install -r requirements.txt
```

For the installation of pybgpstream, check the following library or follow the commands on how I installed it (when I ran the installation the documentation was outdated)
- [BGPStream v2.2](https://bgpstream.caida.org/docs/install/pybgpstream)
Download wandio release from: github https://github.com/LibtraceTeam/wandio/releases/tag/4.2.4-1
    ```shell=
	mkdir ~/src
	cd ~/src
	tar zxf wandio-4.2.4.tar.gz
	cd wandio-4.2.4/	
	./bootstrap.sh (only if you've cloned the source from GitHub)
	./configure
	#installing/updating any additional packages that were needed by either of the above runs until both runs are succesful	
	make
	make install
    ```
2) Once wandio was installed, install libbgpstream 
    ```shell=
    cd ~/src/
    curl -LO https://github.com/CAIDA/libbgpstream/releases/download/v2.2.0/libbgpstream-2.2.0.tar.gz
 	tar zxf libbgpstream-2.2.0.tar.gz
	cd libbgpstream-2.2.0/
	./configure
	#installing/updating any additional packages that gave errors during the above run and run again
 	make
 	make check
 	sudo make install
	sudo ldconfig
    ```

3) Once libbgpstream is installed, pybgpstream can be installed

    ```shell=
    pip install pybgpstream
    ```

### 2. Usage
##### 2.1 Building the graph with GraphBuilder.py, and outputting the paths to bgp_paths_out.txt
```shell
$ python GraphBuilder.py -p peeringdb_20230409.json -o as_org_2023_04_09.jsonl -s '2023-04-09 07:50:00' -e '2023-04-09 08:10:00' --write_paths
$ cp bgp_paths_out.txt Toposcope/bgp_paths_out.txt
$ cp as_org_2023_04_09.jsonl Toposcope/as_org_2023_04_09.jsonl
$ cp peeringdb_20230409.json Toposcope/peeringdb_20230409.json
```
It will give an error when building the graph, don't worry, it is normal as we do not have the relationship inference yet. We also copied the asorg and peeringdb file to toposcope as they will be necessary

##### 2.2 Infering link relationships with Toposcope

```sh
$ cd Toposcope
$ python uniquePath.py -i=<aspaths file> -p=<peeringdb file>
# e.g.python uniquePath.py -i=bgp_paths_out.txt -p=peeringdb_20230409.json
# Output is written to 'aspaths.txt'.
```

**Run AS-Rank algorithm to bootstrap TopoScope**

```sh
$ perl asrank.pl aspaths.txt > asrel.txt
```

**Run Toposcope**

```sh
$ python toposcope.py -o=<ASorg file> -p=<peeringdb file> -d=<temporary storage folder name>
#e.g. python toposcope.py -o=as_org_2023_04_09.jsonl -p=peeringdb_20230409.json -d=tmp/
# Output is written to 'asrel_toposcope.txt'.
$ cp asrel_toposcope.txt ../asrel_toposcope.txt
$ cd ..
```

##### 2.3 Creating graph with relationships
```shell 
$ python GraphBuilder.py -p peeringdb_20230409.json -o as_org_2023_04_09.jsonl -s '2023-04-09 07:50:00' -e '2023-04-09 08:10:00'
```

##### 2.4 Processing the data and saving it to json files
```shell
$ python graph_data.py
```

##### 2.5 Building the final dataset
```shell
$ python gt_parser.py
```

Output file is named dataset.npz
