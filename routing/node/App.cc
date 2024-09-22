#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <vector>
#include <map>
#include <omnetpp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

#include "Packet_m.h"

using namespace omnetpp;

/**
 * Generates traffic for the network.
 */

struct PackageData {
    double delaySum = 0.0;
    double jitterSum = 0.0;
    int pkgCount = 0;
};

class App: public cSimpleModule {
private:
    // configuration
    int myAddress;
    int netId;
    std::string appType;
    std::vector<int> destAddresses;
    cPar *sendIATime;
    cPar *packetLengthBytes;

    ///////////////////////////////////
    std::map<int, PackageData> receivedPkgData; // 存储src节点和对应的PackageData
    std::string log_path;
    std::map<int, int> sendPkgcount; // 记录发送的包数
    std::map<int, double> lastPkgDelay; // 用于记录来自相同发送方数据包的上一次延迟，计算抖动
    ///////////////////////////////////

    // state
    cMessage *generatePacket = nullptr;
    long pkCounter;
    simtime_t lastPacketTime;

    // signals
    simsignal_t endToEndDelaySignal;
    simsignal_t endToEndJitterSignal;

    // 丢包率
    simsignal_t sendCountSignal;
    simsignal_t receiveCountSignal;

public:
    virtual ~App();

protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(App);

App::~App() {
//    cancelAndDelete(generatePacket);
    /////////////////////////////////// 写数据
    EV << "finished." << netId << myAddress << endl;
    std::ofstream file(log_path, std::ios::app);
    if (file.is_open()) {
        for (const auto &kv : receivedPkgData) {
            double avgDelay = kv.second.delaySum / kv.second.pkgCount;
            double avgJitter = kv.second.jitterSum / kv.second.pkgCount;
            file << "src: " << kv.first << ", dst: " << myAddress
                    << ", avgDelay: " << avgDelay << ", avgJitter: "
                    << avgJitter << ", pkgReceiveCount: " <<  kv.second.pkgCount << ", sunDelay: " << kv.second.delaySum << ", sunJitter: " << kv.second.jitterSum << std::endl;
        }
        for (const auto &kv : sendPkgcount) {
            int sendCount = kv.second;
            file << "src: " << myAddress << ", dst: " << kv.first
                    << ", sendPkgCount: " << sendCount << std::endl;
        }
        file.close();
    } else {
        EV << "Failed to open file for writing." << endl;
    }
}

void App::initialize() {
    lastPacketTime = 0.0;
    myAddress = par("address");
    netId = getParentModule()->par("netId");

    appType = par("appType").stringValue();
    packetLengthBytes = &par("packetLength");
    sendIATime = &par("sendIaTime");  // volatile parameter
    pkCounter = 0;

    WATCH(pkCounter);
    WATCH(myAddress);

    const char *destAddressesPar = par("destAddresses");
    cStringTokenizer tokenizer(destAddressesPar);  //自定义的分词器
    const char *token;
    while ((token = tokenizer.nextToken()) != nullptr)
        destAddresses.push_back(atoi(token));

    if (destAddresses.size() == 0)
        throw cRuntimeError(
                "At least one address must be specified in the destAddresses parameter!");

    // 发送包定时器
    if ((appType == "router" || appType == "route_reflector") && destAddresses[0] != -1) {
        EV << "appType = " << appType << endl;
        generatePacket = new cMessage("nextPacket");
        scheduleAt(sendIATime->doubleValue(), generatePacket);
    }

    endToEndDelaySignal = registerSignal("endToEndDelay");
    endToEndJitterSignal = registerSignal("endToEndJitter");

    sendCountSignal = registerSignal("sendCount");
    receiveCountSignal = registerSignal("receiveCount");

    ///////////////////////////////////// 创建文件夹
    std::ostringstream oss;
    const char *homeDir = getenv("HOME"); // 获取家目录路径
    if (homeDir == nullptr) {
        EV << "Failed to get home directory" << endl;
        return;
    }
//    oss << homeDir << "/netowrk_sim/routing/logs/" << netId;
    oss << "logs/" << netId;
    std::string folder_path = oss.str();
    oss << "/pkg_log.txt";
    log_path = oss.str();

    EV << log_path << netId << endl;

    // Create directory if it does not exist
    struct stat info;
    if (stat(folder_path.c_str(), &info) != 0) {
        // Directory does not exist, try to create it
        if (mkdir(folder_path.c_str(), 0777) != 0) {
            std::ostringstream ss;
            ss << "1Failed to create directory: " << folder_path;
            throw cRuntimeError(ss.str().c_str());
        } else {
            EV << "Directory created successfully: " << folder_path << endl;
        }
    } else if (info.st_mode & S_IFDIR) {
        EV << "Directory already exists: " << folder_path << endl;
    } else {
        std::ostringstream ss;
        ss << "2Failed to create directory: " << folder_path;
        throw cRuntimeError(ss.str().c_str());
    }
}

void App::handleMessage(cMessage *msg) {
    if (msg == generatePacket) {
        // Sending packet
        int destAddress = destAddresses[intuniform(0, destAddresses.size() - 1)]; // 随机选择一个目的地网络发送数据包

        char pkname[40];
        snprintf(pkname, sizeof(pkname), "pk-%d-to-%d-#%ld", myAddress,
                destAddress, pkCounter++);
        EV << "generating packet " << pkname << endl;

        Packet *pk = new Packet(pkname);
        pk->setByteLength(packetLengthBytes->intValue());
        pk->setKind(intuniform(0, 7));
        pk->setSrcAddr(myAddress);
        pk->setDestAddr(destAddress);
        send(pk, "out");

        ///////////////////////// 发包计数
        auto it = sendPkgcount.find(destAddress);

        if (it == sendPkgcount.end()) {
            sendPkgcount[destAddress] = 1;
        }else{
            it->second += 1;
        }
        /////////////////////////

        scheduleAt(simTime() + sendIATime->doubleValue(), generatePacket);
        emit(sendCountSignal, 1); // 发送数据包+1
        if (hasGUI())
            getParentModule()->bubble("Generating packet...");
    } else {
        // Handle incoming packet
        Packet *pk = check_and_cast<Packet*>(msg);
        int src = pk->getSrcAddr();

        EV << "received packet " << pk->getName() << " after "
                  << pk->getHopCount() << " delay "
                  << simTime() - pk->getCreationTime() << "hops" << endl;
        simtime_t delay = simTime() - pk->getCreationTime();
//        emit(endToEndDelaySignal, delay);  // 发送数据包端到端的delay数据
        auto it = lastPkgDelay.find(src);

        double jitter;

        if(it == lastPkgDelay.end()){
            lastPkgDelay[src] = delay.dbl();
            jitter = 0.0;
        }else{
            jitter = std::abs(delay.dbl() - it->second);
        }

//        emit(endToEndJitterSignal, jitter); // 发送数据包端到端抖动数据

//        emit(receiveCountSignal, 1); // 接受数据包+1
        it -> second = delay.dbl();

        ////////////////////////////

        auto it2 = receivedPkgData.find(src);

        if (it2 == receivedPkgData.end()) {
            receivedPkgData[src] = PackageData { delay.dbl(), jitter, 1 };
        } else {
            it2->second.delaySum += delay.dbl();
            it2->second.jitterSum += jitter;
            it2->second.pkgCount += 1;
        }
        //////////////////////

        delete pk;

        if (hasGUI())
            getParentModule()->bubble("Arrived!");
    }
}
