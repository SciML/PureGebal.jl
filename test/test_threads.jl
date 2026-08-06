using PureGebal: GebalWorkspace, balance!
using Base.Threads: @threads, nthreads
using LinearAlgebra
using Random
using Test

# A workspace is mutated by every call, so it cannot be shared across concurrent
# calls. The failure is silent: the balanced matrix comes out right regardless,
# because the transform never reads `scale` -- only the recorded transform is
# corrupted, which then breaks unbalance! much later. Pin the contract with a
# test so it cannot regress into "seems to work under threads".
@testset "one workspace per task" begin
    n = 48
    ntask = 4 * max(nthreads(), 1)
    mats = [
        randn(Random.Xoshiro(1000 + i), n, n) .*
            exp10.(rand(Random.Xoshiro(i), -6:6, n, n)) for i in 1:ntask
    ]
    serial = map(mats) do A
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        (B, copy(ws.scale), ws.ilo, ws.ihi)
    end

    ok = fill(false, ntask)
    @threads for i in 1:ntask
        B = copy(mats[i])
        ws = GebalWorkspace(B)          # per-task workspace
        balance!(B, ws)
        ok[i] = B == serial[i][1] && ws.scale == serial[i][2] &&
            ws.ilo == serial[i][3] && ws.ihi == serial[i][4]
    end
    @test all(ok)
end
