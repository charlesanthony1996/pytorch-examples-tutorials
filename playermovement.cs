using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
//using System.Diagnostics;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

public class MovementComponent : MonoBehaviour
{
    [SerializeField] private float speed = 0.1f;
    [SerializeField] private MovementType movementType;

    [SerializeField] private Animator animator;

    private Vector3 moveBy;
    private bool isMoving;
    private bool isJumpingOrFalling;

    // Start is called before the first frame update

    void Start()
    {

    }

    void OnMovement(InputValue input)
    {
        Vector2 inputValue = input.Get<Vector2>();
        moveBy = new Vector3(inputValue.x, 0, inputValue.y);
        AkSoundEngine.PostEvent("Play_Footsteps", gameObject);
    }

    void OnJump(InputValue input)
    {
        if (isJumpingOrFalling)
            return;
        GetComponent<Rigidbody>().AddForce(0, 8, 0, ForceMode.VelocityChange);
        AkSoundEngine.PostEvent("Play_Jump_Sound", gameObject);
    }

    void Update()
    {
        ExecuteMovement();
    }

    void ExecuteMovement()
    {
        isJumpingOrFalling = GetComponent<Rigidbody>().velocity.y < -.035 ||
                             GetComponent<Rigidbody>().velocity.y > 0.00001;

        if (moveBy == Vector3.zero)
            isMoving = false;
        else
            isMoving = true;

        animator.SetBool("walk", isMoving);
        animator.SetBool("jump", isJumpingOrFalling);

        if (movementType == MovementType.TransformBased)
        {
            //transform.position += moveBy * (speed * Time.deltaTime);
            transform.Translate(moveBy * (speed * Time.deltaTime));
        }
        else if (movementType == MovementType.PhysicsBased)
        {
            var rigidbody = this.GetComponent<Rigidbody>();
            rigidbody.AddForce(moveBy * 2, ForceMode.Acceleration);
        }

    }

}