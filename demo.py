class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        n, ans, l = len(nums), 0, 0

        for r in range(n):

            if nums[r] == 0:                       # try to pick current 0

                if k == 0:

                    while nums[l] != 0:
                        l += 1
                    l += 1

                else:
                    k -= 1                       # otherwise pick it and decrement k

            # update ans as max window size till now
            ans = max(ans, r - l + 1)

        return ans
