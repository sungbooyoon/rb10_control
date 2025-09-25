from ikpy.chain import Chain

full_chain = Chain.from_urdf_file("/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf", base_elements=['link0'])

# 필요 없는 조인트/링크를 제외하고 새 체인 구성
reduced_chain = Chain(name="rb10_reduced", links=[
    l for (l, active) in zip(full_chain.links, [False, True, True, True, True, True, True, False]) if active
])
my_chain = reduced_chain

print(my_chain.inverse_kinematics([1, 1, 1]))