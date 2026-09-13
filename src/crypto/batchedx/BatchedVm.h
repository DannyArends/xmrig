/* XMRig — batched AVX-512 RandomX (8-lane), header.
 *
 * Compiled in only when WITH_BATCHEDX=ON (XMRIG_FEATURE_BATCHEDX).
 */

#ifndef XMRIG_BATCHEDVM_H
#define XMRIG_BATCHEDVM_H


#include <cstddef>
#include <cstdint>


namespace xmrig {


class BatchedVm
{
public:
    static constexpr int LANES = 8;             // AVX-512 = 8 x 64-bit lanes
};


} // namespace xmrig


#endif // XMRIG_BATCHEDVM_H
