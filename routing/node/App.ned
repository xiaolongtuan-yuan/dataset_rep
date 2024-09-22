//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 1992-2015 Andras Varga
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

package node;

//
// Application model to generate traffic for the network.
//
simple App
{
    parameters:
        int address;  // local node address
        string destAddresses;  // destination addresses
        string appType;
        volatile double sendIaTime @unit(s) = default(exponential(1s)); // time between generating packets
        volatile int packetLength @unit(byte);  // length of one message (fixed! no "volatile" modifier)
        @display("i=block/browser");
        @signal[endToEndDelay](type="simtime_t");
        @signal[endToEndJitter](type="simtime_t");
        
        @statistic[endToEndDelay](title="end-to-end delay of arrived packets"; unit=s; record=mean,max; interpolationmode=none);
        @statistic[endToEndJitter](title="end-to-end jitter of arrived packets"; unit=s; record=mean,max; interpolationmode=none);
        
        @signal[sendCount](type="long");
        @statistic[sendCount](title="send count"; interpolationmode=none; record=count?);
        @signal[receiveCount](type="long");
        @statistic[receiveCount](title="receive count"; interpolationmode=none; record=count?);
    gates:
        input in;
        output out;
}


