#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <chrono>
#include <future>
#include <print>
#include <ranges>
#include <thread>
#include <vector>

namespace sv = std::views;
namespace sr = std::ranges;

using Mesh = OpenMesh::TriMesh_ArrayKernelT<>;
using FaceIter = Mesh::FaceIter;
using VertexHandle = Mesh::VertexHandle;
using FaceHandle = Mesh::FaceHandle;
using Mat = Eigen::Matrix3f;
using MatX = Eigen::MatrixXf;
using Vec = Eigen::Vector3f;
using VecX = Eigen::VectorXf;
using SpMat = Eigen::SparseMatrix<float>;
using SpMatItem = Eigen::Triplet<float>;

struct MeshFaceHandlesRange
    : std::ranges::view_interface<MeshFaceHandlesRange> {
  const Mesh &mesh;

  MeshFaceHandlesRange(const Mesh &mesh) : mesh(mesh) {}

  auto begin() const { return mesh.faces_begin(); }

  auto end() const { return mesh.faces_end(); }
};

struct TimedScope {
  using clock = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  time_point start;
  std::string name;

  TimedScope(const std::string &name) : name(name) { start = clock::now(); }

  ~TimedScope() {
    using namespace std::chrono;
    auto end = clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start);
    std::println("{} took {} ms", name, elapsed.count());
  }
};

template <typename F, typename V>
void execute_as_async(F &&func, V &&view) {
  const size_t num_workers = std::thread::hardware_concurrency();

  const size_t n = sr::distance(view);
  if (n == 0) return;
  const size_t actual_workers = std::min(num_workers, n);
  const size_t chunk_size = (n + actual_workers - 1) / actual_workers;

  auto futures = view | sv::chunk(chunk_size) |
                 sv::transform([&]<typename C>(C &&chunk) {
                   return std::async(std::launch::async, std::forward<F>(func),
                                     std::forward<C>(chunk));
                 }) |
                 sr::to<std::vector>();

  sr::for_each(futures, [](auto &future) { future.get(); });
}

auto calc_V_matrix(const Mesh &mesh, const FaceHandle &face) {
  std::vector<Vec> vertices;
  vertices.reserve(3);
  for (const auto &vh : mesh.fv_range(face)) {
    auto pos = mesh.point(vh);
    vertices.emplace_back(pos[0], pos[1], pos[2]);
  }

  assert(vertices.size() == 3 && "Not triangular");

  const auto &v1 = vertices[0], &v2 = vertices[1], &v3 = vertices[2];
  const auto d12 = v2 - v1, d13 = v3 - v1;
  auto cross = d12.cross(d13);
  const auto d14 = cross / cross.norm();

  Mat res;
  res << d12, d13, d14;
  return res;
}

Mesh deform_trans(const Mesh &s0, const Mesh &s1, const Mesh &t0) {
  using namespace std::chrono;

  MeshFaceHandlesRange s0_face_handles{s0}, s1_face_handles{s1},
      t0_face_handles{t0};

  if (!(s0.n_vertices() == s1.n_vertices() &&
        s1.n_vertices() == t0.n_vertices() && s0.n_faces() == s1.n_faces() &&
        s1.n_faces() == t0.n_faces())) {
    throw std::invalid_argument("Not same topology");
  }

  const size_t N = s0.n_vertices(), M = s0.n_faces();

  MatX S(3, 3 * M);

  const auto calc_S = [&](const auto &chunk) {
    for (const auto &[j, fh0, fh1] : chunk) {
      auto Vs0j = calc_V_matrix(s0, fh0);
      auto Vs1j = calc_V_matrix(s1, fh1);
      S(Eigen::all, Eigen::seqN(3 * j, 3)) = Vs1j * Vs0j.inverse();
    }
  };

  {
    TimedScope scope{"Calculating S matrix"};
    execute_as_async(calc_S, sv::zip(sv::iota(size_t{0}, M), s0_face_handles,
                                     s1_face_handles));
  }

  std::vector<SpMatItem> A_items;

  std::mutex A_lock;
  const auto calc_A = [&](const auto &chunk) {
    std::vector<int> indices;
    indices.reserve(3);

    for (const auto &[j, fh] : chunk) {
      auto Vt0j = calc_V_matrix(t0, fh);

      indices.clear();
      for (const auto &vh : t0.fv_range(fh)) {
        assert(0 <= vh.idx() && vh.idx() < N && "Unexpected index");
        indices.push_back(vh.idx());
      }

      assert(indices.size() == 3 && "Not triangle");

      std::array items{
          SpMatItem(0, indices[0], -1), SpMatItem(0, indices[1], 1),
          SpMatItem(1, indices[0], -1), SpMatItem(1, indices[2], 1),
          SpMatItem(2, indices[0], -1), SpMatItem(2, N + j, 1),
      };

      SpMat I(3, N + M);
      I.setFromTriplets(items.begin(), items.end());
      SpMat Vt0j_inv = Vt0j.transpose().inverse().sparseView();
      SpMat A_block = Vt0j_inv * I;

      {
        std::lock_guard guard(A_lock);
        for (int k = 0; k < A_block.outerSize(); ++k) {
          for (SpMat::InnerIterator it(A_block, k); it; ++it) {
            A_items.emplace_back(it.row() + 3 * j, it.col(), it.value());
          }
        }
      }
    }
  };

  SpMat A(9 * M, 3 * (N + M));

  {
    TimedScope scope{"Calculating A matrix"};

    execute_as_async(calc_A, sv::enumerate(t0_face_handles));

    const size_t A_size = A_items.size();
    for (const auto i : sv::iota(1, 3)) {
      for (const auto j : sv::iota(size_t{0}, A_size)) {
        const auto &item = A_items[j];
        A_items.emplace_back(item.row() + 3 * i * M, item.col() + i * (N + M),
                             item.value());
      }
    }

    A.setFromTriplets(A_items.begin(), A_items.end());
  }

  auto ATA = A.transpose() * A;

  Eigen::SparseLU<SpMat> solver;
  solver.analyzePattern(ATA);
  solver.factorize(ATA);

  if (solver.info() != 0) {
    throw std::runtime_error("Singular matrix");
  }

  auto t1_vertex_pos = [&]() {
    TimedScope scope{"Solving target vertices"};

    MatX res = solver.solve(A.transpose() * S.transpose().reshaped())
                   .reshaped(N + M, 3)
                   .transpose()
                   .leftCols(N);

    return res;
  }();

  Mesh t1{t0};
  auto t1_vertices = t1.vertices();
  for (const auto &[i, vh] : sv::enumerate(t1_vertices)) {
    const auto &pos = t1_vertex_pos.col(i);
    t1.set_point(vh, {pos(0), pos(1), pos(2)});
  }
  return t1;
}

int main(int argc, char **argv) {
  using namespace OpenMesh::IO;

  try {
    if (argc != 5) {
      throw std::runtime_error(
          "Please specify input s0, s1 and t0, output t1.");
    }
    
    Mesh s0, s1, t0;
    if (!read_mesh(s0, argv[1]) || !read_mesh(s1, argv[2]) ||
        !read_mesh(t0, argv[3])) {
      throw std::runtime_error("Failed to load mesh!");
    }

    Mesh t1 = deform_trans(s0, s1, t0);

    if (!write_mesh(t1, argv[4])) {
      throw std::runtime_error("Failed to save mesh!");
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}