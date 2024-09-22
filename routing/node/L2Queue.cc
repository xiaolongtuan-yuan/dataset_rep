//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 1992-2015 Andras Varga
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>

using namespace omnetpp;

/**
 * Point-to-point interface module. While one frame is transmitted,
 * additional frames get queued up; see NED file for more info.
 */
class L2Queue : public cSimpleModule
{
  private:
    intval_t frameCapacity;

    cQueue queue;
    cMessage *endTransmissionEvent = nullptr;
    bool isBusy;

  public:
    virtual ~L2Queue();

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void refreshDisplay() const override;
    virtual void startTransmitting(cMessage *msg);
};

Define_Module(L2Queue);

L2Queue::~L2Queue()
{
    cancelAndDelete(endTransmissionEvent);
}

void L2Queue::initialize()
{
    queue.setName("queue");
    endTransmissionEvent = new cMessage("endTxEvent");

    if (par("useCutThroughSwitching"))
        gate("line$i")->setDeliverImmediately(true);

    frameCapacity = par("frameCapacity");

    isBusy = false;
}

void L2Queue::startTransmitting(cMessage *msg)
{
    EV << "Starting transmission of " << msg << endl;
    isBusy = true;
    int64_t numBytes = check_and_cast<cPacket *>(msg)->getByteLength();
    send(msg, "line$o");

    // Schedule an event for the time when last bit will leave the gate.
    simtime_t endTransmission = gate("line$o")->getTransmissionChannel()->getTransmissionFinishTime();
    scheduleAt(endTransmission, endTransmissionEvent);
}

void L2Queue::handleMessage(cMessage *msg)
{
    if (msg == endTransmissionEvent) {
        // Transmission finished, we can start next one.
        EV << "Transmission finished.\n";
        isBusy = false;
        if (!queue.isEmpty()) {
            msg = (cMessage *)queue.pop();
            startTransmitting(msg);
        }
    }
    else if (msg->arrivedOn("line$i")) {
        // pass up
        send(msg, "out");
    }
    else {  // arrived on gate "in"
        if (endTransmissionEvent->isScheduled()) {
            // We are currently busy, so just queue up the packet.
            if (frameCapacity && queue.getLength() >= frameCapacity) {
                EV << "Received " << msg << " but transmitter busy and queue full: discarding\n";
                delete msg;
            }
            else {
                EV << "Received " << msg << " but transmitter busy: queueing up\n";
                msg->setTimestamp();
                queue.insert(msg);
            }
        }
        else {
            // We are idle, so we can start transmitting right away.
            EV << "Received " << msg << endl;
            startTransmitting(msg);
        }
    }
}

void L2Queue::refreshDisplay() const
{
    getDisplayString().setTagArg("t", 0, isBusy ? "transmitting" : "idle");
    getDisplayString().setTagArg("i", 1, isBusy ? (queue.getLength() >= 3 ? "red" : "yellow") : "");
}

