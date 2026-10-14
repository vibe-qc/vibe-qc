#include "vibeqc/manta_bubbles.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_manta_bubbles(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<BubbleOptions>(m,"BubbleOptions")
        .def(py::init<>())
        .def_readwrite("leaf_capacity",&BubbleOptions::leaf_capacity)
        .def_readwrite("branching",&BubbleOptions::branching)
        .def_readwrite("maximum_depth",&BubbleOptions::maximum_depth);
    py::class_<BubblePlan>(m,"BubblePlan")
        .def_readonly("pairs",&BubblePlan::pairs)
        .def_readonly("maximum_nodes",&BubblePlan::maximum_nodes)
        .def_readonly("scalar_terms",&BubblePlan::scalar_terms)
        .def_readonly("kernel_terms",&BubblePlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BubblePlan::modeled_peak_bytes);
    py::class_<BubbleNode>(m,"BubbleNode")
        .def_readonly("center",&BubbleNode::center)
        .def_readonly("radius",&BubbleNode::radius)
        .def_readonly("begin",&BubbleNode::begin)
        .def_readonly("end",&BubbleNode::end)
        .def_readonly("parent",&BubbleNode::parent)
        .def_readonly("depth",&BubbleNode::depth)
        .def_readonly("always_near",&BubbleNode::always_near)
        .def_property_readonly("children",[](const BubbleNode& n) { return std::vector<Count>(n.children.begin(),n.children.begin()+n.child_count); });
    py::class_<BubbleBlock>(m,"BubbleBlock")
        .def_readonly("target",&BubbleBlock::target)
        .def_readonly("source",&BubbleBlock::source)
        .def_readonly("pair_interactions",&BubbleBlock::pair_interactions)
        .def_readonly("far_candidate",&BubbleBlock::far_candidate)
        .def_readonly("separation_ratio",&BubbleBlock::separation_ratio);
    py::class_<BubbleTraversalReceipt>(m,"BubbleTraversalReceipt")
        .def_readonly("visited_node_pairs",&BubbleTraversalReceipt::visited_node_pairs)
        .def_readonly("near_blocks",&BubbleTraversalReceipt::near_blocks)
        .def_readonly("far_blocks",&BubbleTraversalReceipt::far_blocks)
        .def_readonly("pair_interactions",&BubbleTraversalReceipt::pair_interactions)
        .def_readonly("refined_candidates",&BubbleTraversalReceipt::refined_candidates)
        .def_readonly("maximum_stack",&BubbleTraversalReceipt::maximum_stack)
        .def_readonly("modeled_peak_bytes",&BubbleTraversalReceipt::modeled_peak_bytes)
        .def_readonly("complete",&BubbleTraversalReceipt::complete)
        .def_readonly("failed",&BubbleTraversalReceipt::failed)
        .def_readonly("source_identity",&BubbleTraversalReceipt::source_identity)
        .def_readonly("partition_identity",&BubbleTraversalReceipt::partition_identity);
    py::class_<BubbleTree,std::shared_ptr<BubbleTree>>(m,"BubbleTree")
        .def(py::init<const PairExtents&,const BubbleOptions&,const ResourceBudget&>(),py::arg("extents"),py::arg("options")=BubbleOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("nodes",[](const BubbleTree& t) { return std::vector<BubbleNode>(t.nodes()); })
        .def_property_readonly("pair_order",[](const BubbleTree& t) { return std::vector<Count>(t.pair_order()); })
        .def_property_readonly("pairs",[](const BubbleTree& t) { return std::vector<PairExtent>(t.pairs()); })
        .def_property_readonly("plan",[](const BubbleTree& t) { return BubblePlan(t.plan()); })
        .def_property_readonly("source_identity",&BubbleTree::source_identity)
        .def_property_readonly("depth",&BubbleTree::depth)
        .def("traversal",[](std::shared_ptr<BubbleTree> t,double opening,const ResourceBudget& budget) { return std::make_unique<BubbleTraversal>(std::move(t),opening,budget); },py::arg("opening")=.5,py::arg("budget")=ResourceBudget{});
    py::class_<BubbleTraversal>(m,"BubbleTraversal")
        .def("peek",&BubbleTraversal::peek)
        .def("accept",&BubbleTraversal::accept)
        .def("refine",&BubbleTraversal::refine)
        .def_property_readonly("receipt",[](const BubbleTraversal& t) { return BubbleTraversalReceipt(t.receipt()); })
        .def("__iter__",[](BubbleTraversal& t)->BubbleTraversal& { return t; },py::return_value_policy::reference_internal)
        .def("__next__",[](BubbleTraversal& t) { auto block=t.peek(); if(!block) throw py::stop_iteration(); t.accept(); return *block; });
    m.def("plan_bubbles",&plan_bubbles,py::arg("extents"),py::arg("options")=BubbleOptions{},py::arg("budget")=ResourceBudget{});
}
