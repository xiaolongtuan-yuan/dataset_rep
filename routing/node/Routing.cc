//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 1992-2015 Andras Varga
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <map>
#include <string>
#include <omnetpp.h>
#include <sstream>
#include "Packet_m.h"

using namespace omnetpp;

/**
 * Demonstrates static routing, utilizing the cTopology class.
 */
class Routing: public cSimpleModule {
private:
    int myAddress;

    std::string routingTableString;
    typedef std::map<int, std::map<int, int>> RoutingTable; // destaddr -> gateindex
    RoutingTable rtable;

protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    void readTopologyFile();
};

Define_Module(Routing);

void Routing::initialize() {
    myAddress = getParentModule()->par("address");

    routingTableString = getParentModule()->par("routingTableString").stringValue();

    //
    // Brute force approach -- every node does topology discovery on its own,
    // and finds routes to all other nodes independently, at the beginning
    // of the simulation. This could be improved: (1) central routing database,
    // (2) on-demand route calculation

    readTopologyFile();
    for (const auto &entry : rtable) {
        int startNode = entry.first;
        const auto &destPorts = entry.second;
        for (const auto &destPort : destPorts) {
            int destNode = destPort.first;
            int port = destPort.second;
            EV << "From Node " << startNode << " to Node " << destNode
                    << " : Port " << port << endl;
        }
    }
}

void Routing::handleMessage(cMessage *msg) {
    Packet *pk = check_and_cast<Packet*>(msg);
    int destAddr = pk->getDestAddr();

    if (destAddr == myAddress) {
        EV << "local delivery of packet " << pk->getName() << endl;
        send(pk, "localOut");
        return;
    }

    // 查找路由表
    auto rtableLine = rtable.find(myAddress);
    if (rtableLine != rtable.end()) {
        const auto &destPorts = rtableLine->second;
        auto portIt = destPorts.find(destAddr);
        if (portIt != destPorts.end()) {
            int outGateIndex = portIt->second;
            EV << "forwarding packet " << pk->getName() << " on gate index "
                      << outGateIndex << endl;
            pk->setHopCount(pk->getHopCount() + 1);

            send(pk, "out", outGateIndex);
            return;
        } else {
            EV << "address " << destAddr << " unreachable, discarding packet "
                      << pk->getName() << endl;
            delete pk;
            return;
        }
    }
}
void Routing::readTopologyFile() {
    std::istringstream iss(routingTableString);
    std::string line;
    int row = 0;
    while (getline(iss, line)) {
        std::stringstream ss(line);
        int col = 0;
        int port;
        while (ss >> port) {
            if (port != -1) {
                rtable[row][col] = port;
            }
            col++;
        }
        row++;
    }
    return;
}

