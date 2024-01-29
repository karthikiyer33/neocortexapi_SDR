// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortex;
using NeoCortexApi;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;

namespace UnitTestsProject
{
    [TestClass]
    public class AdaptSegmentsSEProject
    {


        [TestMethod]

        public void AdaptSegment_DecreasePermanence_SingleSynapse()
        {
            // Arrange
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            tm.Init(cn);

            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(1));
            Synapse synapse = cn.CreateSynapse(dd, cn.GetCell(15), 0.9);

            // Act: Adapt the segment with a single synapse and permanence decrement
            TemporalMemory.AdaptSegment(cn, dd, cn.GetCells(new int[] { }), 0.1, 0.2);

            // Assert: Verify that the synapse's permanence has been appropriately decreased
            Assert.AreEqual(0.7, synapse.Permanence, 0.01);
        }


        [TestMethod]
        public void AdaptSegmmentNoSynapses_DestroyDistalDendriteSegment()
        {

            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            tm.Init(cn);


            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(1));


            TemporalMemory.AdaptSegment(cn, dd, new List<Cell>(), -0.1, -0.05);


            Assert.AreEqual(0, dd.Synapses.Count, "The segment should be destroyed with no synapses.");
        }
    } 
}