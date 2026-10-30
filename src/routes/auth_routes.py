from fastapi import APIRouter, Depends, Response

from src.controllers.auth_controller import (
    RegisterInput,
    LoginInput,
    register_user,
    login_user,
    logout_user,
    get_current_user,
)

router = APIRouter()


@router.post("/register", status_code=201)
async def register(data: RegisterInput, response: Response):
    return await register_user(data, response)


@router.post("/login")
async def login(data: LoginInput, response: Response):
    return await login_user(data, response)


@router.post("/logout")
async def logout(response: Response):
    return await logout_user(response)


@router.get("/me")
async def me(current_user=Depends(get_current_user)):
    return current_user